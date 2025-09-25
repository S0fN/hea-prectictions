import pandas as pd
import math
import numpy as np

class ChemicalEngine:
    def __init__(self, data_folder_path='./data/'):
        print("Initializing Chemical Rules Engine...")
        self.R = 8.314      # J/(mol*K)
        self.F = 96485.0    # C/mol

        try:
            self.salts_df = pd.read_csv(f'{data_folder_path}D1_D4_Master_Salts.csv').rename(columns=str.strip)
            print("DEBUG LOG (from chemical_engine.py): Metals found ->", sorted(self.salts_df['Metal'].unique().tolist()))
            self.pka_df = pd.read_csv(f'{data_folder_path}D7_pKa_pKb_Values.csv').rename(columns=str.strip)
            self.stability_df = pd.read_csv(f'{data_folder_path}D8_Stability_Constants.csv').rename(columns=str.strip)
            self.ion_params_df = pd.read_csv(f'{data_folder_path}D5_Ion_Parameters.csv').rename(columns=str.strip)
            if 'ion_charge_z' in self.ion_params_df.columns:
                self.ion_params_df.rename(columns={'ion_charge_z': 'Charge'}, inplace=True)
            self.ligand_charges_df = pd.read_csv(f'{data_folder_path}D9_Ligand_Charges.csv').rename(columns=str.strip)
            self.additives_df = pd.read_csv(f'{data_folder_path}D6_Additives.csv').rename(columns=str.strip)
            print("All datasets loaded successfully.")

            # --- Pre-cache data into dictionaries for faster lookups ---
            print("Pre-caching data for faster lookups...")
            self._salt_info_cache = {
                (row['Metal'], row['Anion']): row.to_dict()
                for _, row in self.salts_df.iterrows()
            }
            self._ion_params_cache = {
                row.get('IonSymbol', row.get('Metal')): row.to_dict()
                for _, row in self.ion_params_df.iterrows()
            }
            self._stability_cache = {
                (row['Metal'], row['Ligand']): row.to_dict()
                for _, row in self.stability_df.iterrows()
            }
            self._pka_cache = {
                row['CompoundName']: row.to_dict()
                for _, row in self.pka_df.iterrows() if pd.notna(row.get('CompoundName'))
            }
            self._ligand_charge_cache = {
                row['Ligand']: row.to_dict()
                for _, row in self.ligand_charges_df.iterrows()
            }
            print("Pre-caching complete.")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Dataset missing: {e}")

    # ---------------------------
    # Utility Functions
    # ---------------------------

    def select_best_anion(self, metals_list):
        """
        Finds the common anion with the highest minimum solubility across all chosen metals.
        Uses pre-cached salt data for speed.
        """
        if not metals_list:
            return None

        # Get all possible anions for the first metal from the cache keys
        common_anions = {anion for metal, anion in self._salt_info_cache if metal == metals_list[0]}

        # Find the intersection of anions for all other metals
        for metal in metals_list[1:]:
            metal_anions = {anion for m, anion in self._salt_info_cache if m == metal}
            common_anions.intersection_update(metal_anions)

        if not common_anions:
            return None

        best_anion = None
        max_min_sol = -1

        for anion in common_anions:
            min_sol = float('inf')
            for metal in metals_list:
                salt_info = self.get_salt_info(metal, anion)
                sol = salt_info.get('Solubility_g_per_100mL', 0)
                if isinstance(sol, str):  # Handle text values like "Soluble"
                    sol = 200  # Assign a high solubility score
                min_sol = min(min_sol, sol)

            if min_sol > max_min_sol:
                max_min_sol = min_sol
                best_anion = anion
        return best_anion

    def to_kelvin(self, t):
        """Converts temperature to Kelvin if it's in Celsius."""
        return t + 273.15 if t < 100 else t

    def get_salt_info(self, metal, anion):
        """Retrieves all data for a specific metal salt using the fast cache."""
        return self._salt_info_cache.get((metal, anion))

    def get_ion_param(self, ion_name):
        """Retrieves ion parameters using the fast cache."""
        return self._ion_params_cache.get(ion_name)

    def get_stability_constant(self, metal, ligand):
        """Retrieves stability constants using the fast cache."""
        return self._stability_cache.get((metal, ligand))

    def get_pka_info(self, compound_name_part):
        """Finds a compound in the pKa cache by partial name match."""
        for name, data in self._pka_cache.items():
            if compound_name_part in name:
                return data
        return None

    def get_ligand_charge(self, ligand):
        """Retrieves ligand charge using the fast cache."""
        return self._ligand_charge_cache.get(ligand)

    def get_available_ligands(self, metal):
        """Gets a list of ligands known to complex with a given metal."""
        return [key[1] for key in self._stability_cache if key[0] == metal]

    # ---------------------------
    # Chemical Calculation Core
    # ---------------------------

    def calculate_ionic_strength(self, sample, metals, anion):
        """Calculates total ionic strength from all ions in the solution."""
        total_I = 0
        anion_info = self.get_salt_info(metals[0], anion)
        if not anion_info: return 0 # Cannot calculate if anion info is missing
        anion_charge = anion_info.get('AnionCharge', 0)
        total_anion_conc = 0

        for metal in metals:
            ion_param = self.get_ion_param(metal)
            if not ion_param: continue
            charge = ion_param['Charge']
            conc = sample.get(f'conc_{metal}', 0)
            total_I += 0.5 * conc * (charge**2)

            metal_salt_info = self.get_salt_info(metal, anion)
            if metal_salt_info:
                total_anion_conc += conc * abs(metal_salt_info.get('AnionStoich', 1.0))

            ligand = sample.get(f'ligand_{metal}')
            if ligand and ligand != "None":
                lig_charge_info = self.get_ligand_charge(ligand)
                if lig_charge_info:
                    lig_charge = lig_charge_info['Charge']
                    lig_conc = sample.get(f'ligand_conc_{metal}', 0)
                    total_I += 0.5 * (lig_charge**2) * lig_conc

        total_I += 0.5 * total_anion_conc * (anion_charge**2)
        return total_I

    def calculate_activity_coefficient(self, ion_name, ionic_strength):
        """Prefer empirical gamma from D5, else use Extended Debye-Hückel."""
        ion_param = self.get_ion_param(ion_name)
        if not ion_param: return 1.0

        if 'gamma_empirical' in ion_param and pd.notna(ion_param['gamma_empirical']):
            return np.clip(ion_param['gamma_empirical'], 0.1, 1.5)

        charge = ion_param.get('Charge')
        if charge is None or ionic_strength <= 0: return 1.0

        A = 0.509
        term1 = math.sqrt(ionic_strength) / (1 + math.sqrt(ionic_strength))
        term2 = 0.3 * ionic_strength
        log_gamma = -A * (charge**2) * (term1 - term2)
        return max(0.1, min(10**log_gamma, 1.5))

    def calculate_potential_with_ligand(self, metal, salt_info, conc, ligand_name, lig_conc, temp, ionic_strength):
        """Calculates the final potential including Nernst and ligand effects."""
        E0 = salt_info['E0_V_vs_SHE']
        n = salt_info['n_electrons']
        gamma = self.calculate_activity_coefficient(metal, ionic_strength)
        activity = gamma * conc

        if activity <= 1e-12: return np.nan
        temp_K = self.to_kelvin(temp)
        base_E = E0 + (self.R * temp_K) / (n * self.F) * math.log(activity)

        if ligand_name and ligand_name != "None" and lig_conc > 0:
            stab_info = self.get_stability_constant(metal, ligand_name)
            if stab_info:
                log_beta = stab_info['log_beta']
                p = stab_info['Stoichiometry_m']
                if lig_conc <= 1e-9: return base_E # Avoid math domain error with tiny ligand conc
                shift = -(self.R * temp_K / (n * self.F)) * (log_beta * 2.303 + p * math.log(lig_conc))
                return base_E + shift
        return base_E

    def calculate_precipitation_limit(self, metal, sample, metals, anion):
        """Calculates the maximum metal activity before hydroxide precipitation."""
        pka_info = self.get_pka_info(f"{metal}(")
        if not pka_info or pd.isna(pka_info.get('Ksp')): return None

        Ksp = pka_info['Ksp']
        pH = sample['bath_ph']
        ionic_strength = self.calculate_ionic_strength(sample, metals, anion)
        gamma_OH = self.calculate_activity_coefficient('OH', ionic_strength)
        
        pOH = 14.0 - pH
        conc_OH = 10**(-pOH)
        activity_OH = gamma_OH * conc_OH
        if activity_OH <= 1e-12: return None
        
        ion_param = self.get_ion_param(metal)
        if not ion_param: return None
        metal_charge = ion_param['Charge']
        return Ksp / (activity_OH ** metal_charge)

    def calculate_final_potential(self, metal, sample, anion, ionic_strength):
        """Calculates the final potential for one metal, checking for precipitation."""
        salt_info = self.get_salt_info(metal, anion)
        if not salt_info: return np.nan
        
        conc = sample.get(f'conc_{metal}', 0)
        max_activity = self.calculate_precipitation_limit(metal, sample, [metal], anion)
        gamma = self.calculate_activity_coefficient(metal, ionic_strength)
        current_activity = gamma * conc
        if max_activity is not None and current_activity > max_activity:
            return np.nan # Recipe is invalid due to precipitation

        potential = self.calculate_potential_with_ligand(
            metal, salt_info, conc, sample.get(f'ligand_{metal}'),
            sample.get(f'ligand_conc_{metal}', 0), sample['temperature'], ionic_strength
        )
        
        overpotential = sample.get(f'overpotential_{metal}', 0)
        return potential + overpotential if pd.notna(potential) else np.nan

