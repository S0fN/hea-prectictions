import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
import re
import json
import os
from dotenv import load_dotenv
from chemical_engine import ChemicalEngine
import google.generativeai as genai

# --- Page & API Configuration ---
st.set_page_config(page_title="HEA Recipe Optimizer", page_icon="🧪", layout="wide")
load_dotenv()
try:
    GEMINI_API_KEY = "AIzaSyA_crXkTaXif25kWq9tn86rTEkb432O_fs"
    if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY not found.")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"API key configuration error: {e}", icon="🚨")

# --- Asset Caching ---
@st.cache_resource
def load_assets():
    """Loads all necessary assets for the application."""
    try:
        engine = ChemicalEngine(data_folder_path='./data/')
        reg_model = joblib.load('hea_regressor.joblib')
        clf_model = joblib.load('hea_classifier.joblib')
        platable_metals = sorted(engine.salts_df['Metal'].unique().tolist())

        st.info(f"DEBUG INFO (from app.py): The following {len(platable_metals)} metals were loaded from the data file: {platable_metals}", icon="🔬")
        if 'Cd' not in platable_metals:
            st.error("CONFIRMED: 'Cd' is missing from the data loaded by the live application.", icon="🚨")

        try:
            success_df = pd.read_csv('final_stratified_dataset_5metals.csv')
        except FileNotFoundError:
            st.warning("`final_stratified_dataset_5metals.csv` not found. Data-driven fallback will be disabled.")
            success_df = None
        return engine, reg_model, clf_model, platable_metals, success_df
    except Exception as e:
        st.error(f"Error loading assets: {e}. Please ensure you've run the training script and all data files are present.", icon="🚨")
        return None, None, None, [], None

# --- Analysis & Fallback Logic ---
def find_best_fallback_combination(failed_metals, success_df):
    """Analyzes the success dataset to find the best alternative metal combination."""
    if success_df is None: return None
    metal_cols = [col for col in success_df.columns if col.startswith('metal_')]
    success_df['combination'] = success_df[metal_cols].apply(
        lambda row: tuple(sorted([col.replace('metal_', '') for col, val in row.items() if val == 1])),
        axis=1
    )
    combination_counts = success_df['combination'].value_counts().reset_index()
    combination_counts.columns = ['combination', 'count']
    failed_set = set(failed_metals)
    combination_counts['similarity'] = combination_counts['combination'].apply(
        lambda combo: len(failed_set.intersection(set(combo)))
    )
    combination_counts = combination_counts.sort_values(by=['similarity', 'count'], ascending=[False, False])
    if not combination_counts.empty:
        best_fallback = combination_counts.iloc[0]['combination']
        if set(best_fallback) != failed_set:
            return list(best_fallback)
    return None

# --- Core Functions (LLM Calls, Search, Validation) ---
def call_llm_for_report(recipe_data, metals, user_goal):
    st.info("Generating final analysis report...", icon="🤖")
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt_recipe = recipe_data.copy()
    keys_to_remove = [k for k in prompt_recipe.keys() if 'prob' in k or 'predicted' in k or 'metal_' in k or 'anion_' in k or 'ligand_' in k or 'actual_delta_E' in k]
    for key in keys_to_remove: prompt_recipe.pop(key, None)
    
    final_de = recipe_data['actual_delta_E']

    base_prompt = f"""
    As an expert electrochemist specializing in High Entropy Alloys (HEAs), analyze the following recipe.
    **Crucial Context:** The primary goal of single-step HEA electroplating is to bring the deposition potentials of all constituent metals as close together as possible. A **very low final ΔE (ideally < 0.15V) is the main indicator of a successful recipe**.
    **User's Goal:** "{user_goal}"
    **Selected Metals:** {metals}
    **Final Recipe Data:**
    ```json
    {json.dumps(prompt_recipe, indent=2)}
    ```
    """

    if final_de > 0.8:
        analysis_prompt = f"""
        **Analysis Task:**
        The final ΔE of {final_de:.4f} V is extremely high, indicating a failed recipe. Please provide an analysis explaining why co-deposition is likely **fundamentally impossible or extremely difficult** for this combination of metals. Focus on:
        1.  **Electrochemical Incompatibility:** Analyze the standard reduction potentials (E0) and explain if the initial gap is too large to be closed by ligands or concentration effects.
        2.  **Chemical Incompatibility:** Are there other known issues, like competing reactions or lack of suitable complexing agents for this specific group?
        3.  **Conclusion:** State clearly that this combination is not recommended for single-step co-deposition.
        """
    elif 0.2 < final_de <= 0.8:
        analysis_prompt = f"""
        **Analysis Task:**
        The final ΔE of {final_de:.4f} V is high, indicating the ML-guided search failed to find a viable recipe. Please explain the likely reasons for this failure.
        1.  **Prediction Mismatch:** Why might the ML models have ranked this as a promising candidate, while the physics engine revealed it to be a poor recipe?
        2.  **"Needle in a Haystack" Problem:** Is it possible a solution exists but is extremely rare and was missed by the search?
        3.  **Missing Chemistry:** Does the failure suggest that the necessary ligands or additives to make this combination work are not present in the system's known data?
        4.  **Conclusion:** Summarize why the search failed for this combination.
        """
    elif 0.1 <= final_de <= 0.2:
        analysis_prompt = f"""
        **Analysis Task:**
        The final ΔE of {final_de:.4f} V is a **promising near-miss** but doesn't meet the ideal target (< 0.1V). Please provide concrete, actionable suggestions for improvement.
        1.  **Identify Outliers:** Look at the `Final Potential (V vs SHE)` for each metal and identify the one or two metals that are the biggest outliers.
        2.  **Suggest Adjustments:** For the outlier metals, suggest specific changes. For example: "Increase the concentration of Ni slightly" or "Try using Citrate as a ligand for Fe to shift its potential more negatively."
        3.  **Fine-tuning:** Suggest small adjustments to the overall pH or temperature that might help close the final small gap.
        4.  **Conclusion:** Summarize the recommended next steps for refining this recipe in a lab.
        """
    else: # Success case
        analysis_prompt = f"""
        **Analysis Task:**
        The final ΔE of {final_de:.4f} V is excellent and indicates a highly viable recipe. Please provide a positive analysis based on the crucial context provided above.
        1.  **Executive Summary:** Confirm that this recipe is highly viable due to its extremely low ΔE value and explain why this is critical for forming a homogenous HEA.
        2.  **Component Roles:** Explain how the specific concentrations and ligands successfully aligned the deposition potentials.
        3.  **Operating Conditions:** Comment on why the chosen pH and temperature are suitable for this system.
        4.  **Next Steps:** What are the key things to verify in a lab experiment to confirm this excellent theoretical result?
        """
    
    prompt = base_prompt + analysis_prompt
    try:
        response = model.generate_content(prompt); return response.text
    except Exception: return "Failed to generate final analysis report."

def call_llm_for_recipe_generation(failed_recipe, metals):
    st.info("Attempting a final, expert-guided recipe generation...", icon="🧠")
    model = genai.GenerativeModel('gemini-1.5-flash')
    json_schema = {"type": "object", "properties": {"bath_ph": {"type": "number"}, "temperature_C": {"type": "number"}, "components": {"type": "array", "items": {"type": "object", "properties": {"metal": {"type": "string"}, "concentration_M": {"type": "number"}, "ligand": {"type": "string"}, "ligand_concentration_M": {"type": "number"}}, "required": ["metal", "concentration_M", "ligand", "ligand_concentration_M"]}}, "reasoning": {"type": "string"}}, "required": ["bath_ph", "temperature_C", "components", "reasoning"]}
    prompt = f"""
    As an expert electrochemist, generate a final, complete recipe for a High Entropy Alloy with a ΔE < 0.12V.
    The metals are: {metals}. An initial search found a promising but unsuccessful recipe with a ΔE of {failed_recipe['actual_delta_E']:.4f} V.
    Here is the data for that failed recipe:
    ```json
    {json.dumps(failed_recipe, indent=2)}
    ```
    Analyze the `Final Potential (V vs SHE)` for each metal to identify outliers. Propose a single, new, complete recipe you believe will succeed.
    - Adjust pH, temperature, and concentrations.
    - **Crucially, the `temperature_C` must be within a practical range for an aqueous bath, between 20 and 60 degrees Celsius.**
    - You MUST choose a ligand for the outlier metals from this list: [Citrate, EDTA, Thiosulfate] or "None".
    - Provide a brief `reasoning` for your changes.
    Return your answer in a strict JSON format matching the schema. Do not add any other text or markdown.
    """
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json", "response_schema": json_schema}); return json.loads(response.text)
    except Exception: return None

def iterative_refinement_search(engine, reg_model, clf_model, metals, anion, num_candidates, top_n_to_validate):
    all_platable_metals = engine.salts_df['Metal'].unique()
    candidates = []
    e0_vals = [engine.get_salt_info(m, anion)['E0_V_vs_SHE'] for m in metals]
    base_features = {'E0_mean': np.mean(e0_vals), 'E0_range': np.ptp(e0_vals), f'anion_{anion}': 1}
    num_initial_random = int(num_candidates * 0.4)
    for _ in range(num_initial_random):
        sample, total_metal_conc = base_features.copy(), 0
        for m in metals:
            sample[f'conc_{m}'] = random.uniform(0.01, 1.2); total_metal_conc += sample[f'conc_{m}']
            available_ligands = engine.get_available_ligands(m)
            if available_ligands and random.random() < 0.7:
                ligand_name = random.choice(available_ligands)
                sanitized_ligand_name = re.sub(r"[^A-Za-z0-9_]+", "", ligand_name)
                sample[f'ligand_{sanitized_ligand_name}'] = 1; sample[f'ligand_conc_{m}'] = random.uniform(0.1, 1.5)
            else: sample[f'ligand_conc_{m}'] = 0
            sample[f'overpotential_{m}'] = random.uniform(-0.45, 0.0)
        sample['bath_ph'], sample['temperature'] = random.uniform(2.0, 9.0), random.uniform(298.15, 333.15)
        sample['total_metal_conc'] = total_metal_conc
        for m in all_platable_metals: sample[f'metal_{m}'] = 1 if m in metals else 0
        candidates.append(sample)
    candidates_df = pd.DataFrame(candidates).fillna(0)
    clf_cols, reg_cols = clf_model.feature_name_, reg_model.feature_name_
    inference_df_clf = candidates_df.reindex(columns=clf_cols, fill_value=0)
    probabilities = clf_model.predict_proba(inference_df_clf)[:, 1]
    candidates_df['feasibility_prob'] = probabilities
    inference_df_reg = candidates_df.reindex(columns=reg_cols, fill_value=0)
    predicted_delta_e = reg_model.predict(inference_df_reg)
    candidates_df['predicted_delta_E'] = predicted_delta_e
    top_parents = candidates_df.nlargest(10, 'feasibility_prob').nsmallest(10, 'predicted_delta_E')
    num_children_per_parent = int((num_candidates * 0.6) / 10)
    if not top_parents.empty:
        for _, parent_series in top_parents.iterrows():
            parent = parent_series.to_dict()
            for _ in range(num_children_per_parent):
                child = parent.copy()
                for m in metals:
                    child[f'conc_{m}'] = np.clip(parent[f'conc_{m}']*(1+random.uniform(-0.1,0.1)), 0.01, 1.5)
                    child[f'ligand_conc_{m}'] = np.clip(parent[f'ligand_conc_{m}']*(1+random.uniform(-0.1,0.1)), 0, 1.5)
                child['bath_ph'] = np.clip(parent['bath_ph']*(1+random.uniform(-0.1,0.1)), 2.0, 9.0)
                candidates.append(child)
    final_candidates_df = pd.DataFrame(candidates).fillna(0)
    final_inference_df_clf = final_candidates_df.reindex(columns=clf_cols, fill_value=0)
    probabilities = clf_model.predict_proba(final_inference_df_clf)[:, 1]
    final_candidates_df['feasibility_prob'] = probabilities
    promising_candidates = final_candidates_df[final_candidates_df['feasibility_prob'] > 0.5].copy()
    if promising_candidates.empty: promising_candidates = final_candidates_df.nlargest(top_n_to_validate, 'feasibility_prob').copy()
    if promising_candidates.empty: return None
    final_inference_df_reg = promising_candidates.reindex(columns=reg_cols, fill_value=0)
    predicted_delta_e = reg_model.predict(final_inference_df_reg)
    promising_candidates['predicted_delta_E'] = predicted_delta_e
    best_candidates_ml = promising_candidates.nsmallest(top_n_to_validate, 'predicted_delta_E')
    return validate_candidates(engine, best_candidates_ml, metals, anion)

def validate_candidates(engine, candidates_df, metals, anion):
    validation_results = []
    for _, candidate_series in candidates_df.iterrows():
        candidate, engine_candidate, potentials = candidate_series.to_dict(), candidate_series.to_dict(), {}
        for m in metals:
            found_ligand, ligand_name = False, "None"
            for col in candidates_df.columns:
                if col.startswith('ligand_') and not col.startswith('ligand_conc_') and candidate.get(col) == 1:
                    sanitized_name = col.replace('ligand_', '')
                    for orig_lig in engine.get_available_ligands(m):
                        if re.sub(r"[^A-Za-z0-9_]+", "", orig_lig) == sanitized_name:
                            ligand_name, found_ligand = orig_lig, True; break
                    if found_ligand: break
            engine_candidate[f'ligand_{m}'] = ligand_name
        ionic_strength = engine.calculate_ionic_strength(engine_candidate, metals, anion)
        for m in metals: potentials[m] = engine.calculate_final_potential(m, engine_candidate, anion, ionic_strength)
        valid_potentials = [p for p in potentials.values() if p is not None and np.isfinite(p)]
        actual_delta_e = np.ptp(valid_potentials) if len(valid_potentials) == 5 else float('inf')
        candidate['actual_delta_E'], candidate['potentials'] = actual_delta_e, potentials
        validation_results.append(candidate)
    if not validation_results: return None
    finite_results = [r for r in validation_results if np.isfinite(r['actual_delta_E'])]
    if not finite_results:
        return min(validation_results, key=lambda x: x.get('predicted_delta_E', float('inf')))

    best_recipe = min(finite_results, key=lambda x: x['actual_delta_E']); 
    best_recipe['anion'] = anion
    return best_recipe

# --- Main Application UI ---
def main():
    engine, reg_model, clf_model, platable_metals, success_df = load_assets()
    if not all([engine, reg_model, clf_model]):
        st.stop()

    st.sidebar.title("🔬 Search Controls")
    num_candidates = st.sidebar.slider("Candidates per Search Stage", 50000, 500000, 150000, 25000)
    top_n_to_validate = st.sidebar.slider("Candidates for Final Validation", 10, 200, 50, 10)

    st.title("🧪 Automated HEA Recipe Optimizer")
    st.markdown("This tool uses a multi-stage, automated process to find viable HEA recipes. It combines ML-guided search, data-driven fallbacks, and expert-guided generation.")

    with st.form("recipe_form"):
        goal_prompt = st.text_input("Describe the desired properties of your alloy", "A tough, corrosion-resistant coating for steel parts")
        selected_metals = st.multiselect("Choose your initial 5 metals.", options=platable_metals, max_selections=5)
        submit_button = st.form_submit_button("Generate Optimal Recipe")

    if submit_button:
        if len(selected_metals) != 5:
            st.warning("Please select exactly 5 metals."); return
        
        final_recipe = None
        current_metals = sorted(selected_metals)
        
        with st.status("Running Automated Optimization Process...", expanded=True) as status:
            status.update(label=f"Stage 1: Running ML-guided search for `{' - '.join(current_metals)}`...")
            anion = engine.select_best_anion(current_metals)
            if not anion:
                status.update(label="Optimization Failed", state="error", expanded=False)
                st.error(f"Could not find a common anion for the selected metals."); return
            
            recipe = iterative_refinement_search(engine, reg_model, clf_model, current_metals, anion, num_candidates, top_n_to_validate)
            
            if recipe and recipe['actual_delta_E'] <= 0.15:
                status.update(label="Success!", state="complete", expanded=False)
                final_recipe = recipe
            else:
                status.update(label=f"Initial search failed (ΔE > 0.15V). Attempting data-driven fallback...")
                fallback_metals = find_best_fallback_combination(current_metals, success_df)
                
                if fallback_metals and set(fallback_metals) != set(current_metals):
                    status.update(label=f"Stage 2: Found promising fallback: `{' - '.join(fallback_metals)}`. Running new search...")
                    fallback_anion = engine.select_best_anion(fallback_metals)
                    recipe = iterative_refinement_search(engine, reg_model, clf_model, fallback_metals, fallback_anion, num_candidates, top_n_to_validate)
                    if recipe and recipe['actual_delta_E'] <= 0.15:
                        status.update(label="Success on Fallback!", state="complete", expanded=False)
                        final_recipe = recipe
                    else:
                        status.update(label="Fallback search failed. Proceeding to final attempt...")
                else:
                    status.update(label="No suitable fallback found. Proceeding to final attempt...")

                if not final_recipe:
                    status.update(label="Stage 3: Asking expert system to generate a final recipe...")
                    best_near_miss = recipe 
                    llm_prompt_recipe = {'actual_delta_E': best_near_miss['actual_delta_E'], 'components': []}
                    for m, p in best_near_miss['potentials'].items():
                        if np.isfinite(p): llm_prompt_recipe['components'].append({'metal': m, 'Final Potential (V vs SHE)': p})
                    
                    llm_suggestion = call_llm_for_recipe_generation(llm_prompt_recipe, sorted(best_near_miss['potentials'].keys()))
                    
                    if llm_suggestion:
                        status.update(label="Validating expert-generated recipe...")
                        engine_candidate = {}
                        llm_metals = [comp['metal'] for comp in llm_suggestion['components']]
                        anion = engine.select_best_anion(llm_metals)
                        if anion: # Check if a common anion exists for the LLM's suggestion
                            engine_candidate['bath_ph'] = llm_suggestion['bath_ph']
                            engine_candidate['temperature'] = engine.to_kelvin(llm_suggestion['temperature_C'])
                            for comp in llm_suggestion['components']:
                                m = comp['metal']
                                engine_candidate[f'conc_{m}'] = comp['concentration_M']
                                engine_candidate[f'ligand_{m}'] = comp['ligand'] if comp['ligand'] != "None" else None
                                engine_candidate[f'ligand_conc_{m}'] = comp['ligand_concentration_M']
                                engine_candidate[f'overpotential_{m}'] = best_near_miss.get(f'overpotential_{m}', -0.1)
                            
                            candidate_df = pd.DataFrame([engine_candidate])
                            final_recipe = validate_candidates(engine, candidate_df, llm_metals, anion)
                            if final_recipe and final_recipe['actual_delta_E'] <= 0.15:
                                status.update(label="Success with Expert-Generated Recipe!", state="complete", expanded=False)
                            else:
                                 status.update(label="All attempts failed to find a viable recipe.", state="error", expanded=False)
                                 final_recipe = best_near_miss
                        else:
                            status.update(label="Expert-generated recipe was chemically invalid (no common anion). All attempts failed.", state="error", expanded=False)
                            final_recipe = best_near_miss
                    else:
                        status.update(label="All attempts failed.", state="error", expanded=False)
                        final_recipe = best_near_miss
        
        st.header("Final Optimization Result")
        if final_recipe and final_recipe['actual_delta_E'] <= 0.15:
            st.success(f"Excellent recipe found with Final ΔE = {final_recipe['actual_delta_E']:.4f} V", icon="🎉")
        elif final_recipe and np.isfinite(final_recipe['actual_delta_E']):
            st.error(f"Could not find a recipe with ΔE ≤ 0.15V. Displaying best attempt: ΔE = {final_recipe['actual_delta_E']:.4f} V")
        else:
            st.error("The optimization process failed to produce any valid recipe. All attempts resulted in physically impossible combinations.")
            if final_recipe: 
                 with st.expander("Show Detailed Numerical Results of Best Invalid Attempt", expanded=True):
                    col1, col2 = st.columns(2)
                    col1.metric("Bath pH", f"{final_recipe.get('bath_ph', 'N/A'):.2f}")
                    col2.metric("Temperature", f"{final_recipe.get('temperature', 298.15)-273.15:.1f} °C")
                    composition_data, potentials_data = [], []
                    recipe_metals = sorted(final_recipe['potentials'].keys())
                    for m in recipe_metals:
                        ligand_display = "None"
                        for col, val in final_recipe.items():
                            if isinstance(col, str) and col.startswith('ligand_') and not col.startswith('ligand_conc_') and val==1:
                                sanitized_ligand_name = col.replace(f'ligand_', '')
                                for orig_lig in engine.get_available_ligands(m):
                                    if re.sub(r"[^A-Za-z0-9_]+", "", orig_lig) == sanitized_ligand_name:
                                        ligand_display = orig_lig; break
                                if ligand_display != "None": break
                        composition_data.append({"Metal": m, "Concentration (M)": final_recipe.get(f'conc_{m}', 0),"Ligand": ligand_display,"Ligand Conc. (M)": final_recipe.get(f'ligand_conc_{m}', 0)})
                        potential_val = final_recipe['potentials'].get(m, float('nan'))
                        potentials_data.append({"Metal": m, "Final Potential (V vs SHE)": f"{potential_val:.4f}" if np.isfinite(potential_val) else "Invalid"})
                    st.subheader(f"Bath Composition for `{' - '.join(recipe_metals)}`")
                    st.dataframe(pd.DataFrame(composition_data).set_index("Metal"))
                    st.subheader("Electrochemical Potentials")
                    st.dataframe(pd.DataFrame(potentials_data).set_index("Metal"))
            return

        with st.expander("Show Detailed Numerical Results", expanded=True):
            col1, col2 = st.columns(2)
            col1.metric("Bath pH", f"{final_recipe.get('bath_ph', 'N/A'):.2f}")
            col2.metric("Temperature", f"{final_recipe.get('temperature', 298.15)-273.15:.1f} °C")
            composition_data, potentials_data = [], []
            recipe_metals = sorted(final_recipe['potentials'].keys())
            for m in recipe_metals:
                ligand_display = "None"
                for col, val in final_recipe.items():
                    if isinstance(col, str) and col.startswith('ligand_') and not col.startswith('ligand_conc_') and val==1:
                        sanitized_ligand_name = col.replace(f'ligand_', '')
                        for orig_lig in engine.get_available_ligands(m):
                            if re.sub(r"[^A-Za-z0-9_]+", "", orig_lig) == sanitized_ligand_name:
                                ligand_display = orig_lig; break
                        if ligand_display != "None": break
                composition_data.append({"Metal": m, "Concentration (M)": final_recipe.get(f'conc_{m}', 0),"Ligand": ligand_display,"Ligand Conc. (M)": final_recipe.get(f'ligand_conc_{m}', 0)})
                potential_val = final_recipe['potentials'].get(m, float('nan'))
                potentials_data.append({"Metal": m, "Final Potential (V vs SHE)": f"{potential_val:.4f}" if np.isfinite(potential_val) else "Invalid"})
            st.subheader(f"Bath Composition for `{' - '.join(recipe_metals)}`")
            st.dataframe(pd.DataFrame(composition_data).set_index("Metal"))
            st.subheader("Electrochemical Potentials")
            st.dataframe(pd.DataFrame(potentials_data).set_index("Metal"))
        
        st.header("Final Expert Analysis")
        report = call_llm_for_report(final_recipe, sorted(final_recipe['potentials'].keys()), goal_prompt)
        st.markdown(report)

if __name__ == "__main__":
    main()
