"""Streamlit UCS prediction and SHAP explanation. © 2026 IA Tijani, IA Lawal & collaborators."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'ModelsGUI'
FEATURES = ['Category', 'VP (m/s)', 'PLS (MPa)', 'SH', 'SH type']
ROCKS = {'Igneous': 1, 'Sedimentary': 2, 'Metamorphic': 3}
SH_TYPES = {'L': 1, 'N': 2}
LIMITS = {'VP (m/s)': (375., 7978.), 'PLS (MPa)': (.36, 23.1), 'SH': (10., 72.)}
DISCLAIMER = 'Model applicability is limited to the multivariate training domain, even when individual inputs fall within their respective ranges.'
st.set_page_config(page_title='UCS | Hybrid Intelligence', page_icon='🪨', layout='wide')

@st.cache_resource(show_spinner='Loading UCS prediction model…')
def artifacts():
    model = joblib.load(ASSETS / 'Best_XGBoost_Regressor_UCS.joblib')
    poly = joblib.load(ASSETS / 'PolyTransformer_UCS.joblib')
    scaler = joblib.load(ASSETS / 'Scaler_UCS.joblib')
    if getattr(poly, 'feature_names_in_', []).tolist() != FEATURES:
        raise ValueError('Saved polynomial feature order differs from application input order.')
    if poly.n_features_in_ != len(FEATURES):
        raise ValueError('Polynomial feature count mismatch.')
    if scaler.n_features_in_ != poly.n_output_features_:
        raise ValueError('Scaler/polynomial dimensions mismatch.')
    if getattr(model, 'n_features_in_', scaler.n_features_in_) != scaler.n_features_in_:
        raise ValueError('Model/scaler dimensions mismatch.')
    return model, poly, scaler

@st.cache_data(show_spinner='Loading SHAP training reference…')
def reference():
    df = pd.read_excel(ASSETS / 'Data_Split_UCS.xlsx', sheet_name='Data_all_Train_70')
    if any(x not in df.columns for x in FEATURES):
        raise ValueError('Training reference workbook lacks required input columns.')
    df = df[FEATURES].apply(pd.to_numeric, errors='coerce').dropna()
    return df[df['Category'].isin(ROCKS.values())].reset_index(drop=True)

def preprocess(df, poly, scaler):
    df = df.reindex(columns=FEATURES).astype(float)
    transformed = poly.transform(df)
    scaled = scaler.transform(transformed)
    return pd.DataFrame(scaled, columns=poly.get_feature_names_out(FEATURES))

def local_domain_distance(point, df):
    """Descriptive nearest-neighbor distance; NOT a calibrated domain or uncertainty test."""
    cols = ['VP (m/s)', 'PLS (MPa)', 'SH']
    ref = df[cols].to_numpy(float)
    q = point[cols].to_numpy(float)
    span = np.ptp(ref, axis=0)
    span[span == 0] = 1
    return float(np.min(np.sqrt(np.sum(((ref-q)/span)**2, axis=1))))

def shap_chart(model, poly, scaler, row, refs, n_reference):
    import shap
    # Preserve saved preprocessing. TreeExplainer explains engineered, scaled inputs.
    refs = refs.sample(n=min(n_reference, len(refs)), random_state=42) if len(refs) > n_reference else refs
    xref = preprocess(refs, poly, scaler)
    xp = preprocess(row, poly, scaler)
    explainer = shap.TreeExplainer(model)
    ref_sv = np.asarray(explainer.shap_values(xref), dtype=float)
    point_sv = np.asarray(explainer.shap_values(xp), dtype=float).reshape(1, -1)
    powers = np.asarray(poly.powers_)
    shares = (powers > 0).astype(float)
    count = shares.sum(axis=1)
    nz = count > 0
    shares[nz] /= count[nz, None]
    # Attribution of polynomial interactions is shared equally among involved raw inputs.
    # Category is omitted visually; its contribution is NOT removed from prediction.
    selected = [1, 2, 3, 4]
    rv = (ref_sv @ shares)[:, selected]
    pv = (point_sv @ shares)[0, selected]
    labels = [FEATURES[i] for i in selected]
    order = np.argsort(np.mean(np.abs(rv), axis=0))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 9))
    parts = ax.violinplot([rv[:, i] for i in order], positions=range(len(order)), vert=False,
                          showmedians=True, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor('#81b3d2'); body.set_alpha(.65)
    ax.scatter(pv[order], range(len(order)), c='#c33434', marker='D', label='Current input', zorder=5)
    ax.set_yticks(range(len(order)), [labels[i] for i in order])
    ax.set_xlabel('SHAP contribution to UCS prediction (MPa)')
    ax.set_title(f'SHAP reference distribution — {len(refs)} {st.session_state.get("rock", "")} samples')
    ax.legend(); ax.axvline(0, color='gray', lw=.8); ax.grid(alpha=.2, axis='x')
    order2 = np.argsort(np.abs(pv))
    ax2.barh(range(len(order2)), pv[order2], color=['#cc5656' if x >= 0 else '#4689b5' for x in pv[order2]])
    ax2.set_yticks(range(len(order2)), [labels[i] for i in order2])
    ax2.set_xlabel('Allocated SHAP contribution (MPa)')
    ax2.set_title('Local SHAP explanation (Category not displayed)')
    ax2.axvline(0, color='gray', lw=.8); ax2.grid(alpha=.2, axis='x')
    fig.tight_layout()
    return fig

st.title('🪨 ML-Aided Uniaxial Compressive Strength (UCS) Prediction')
st.caption('XGBoost regression · Rock-type-specific SHAP analysis · Research demonstration')
st.warning('**Model applicability disclaimer:** ' + DISCLAIMER)
st.info('Inputs are reordered to match the saved training pipeline: Category → VP → PLS → SH → SH type. Individual-range checks do not establish multivariate applicability.')

try:
    model, poly, scaler = artifacts()
except Exception as exc:
    st.error(f'Model files could not be loaded: {exc}')
    st.stop()

with st.form('ucs_form'):
    st.subheader('Rock properties')
    c1, c2 = st.columns(2)
    with c1:
        rock = st.selectbox('Rock category', list(ROCKS), key='rock')
        vp = st.number_input('VP (m/s)', min_value=375.0, max_value=7978.0, value=3000.0, step=10.0)
        pls = st.number_input('PLS (MPa)', min_value=.36, max_value=23.1, value=5.0, step=.1)
    with c2:
        sh = st.number_input('SH', min_value=10.0, max_value=72.0, value=35.0, step=.5)
        shtype = st.selectbox('SH type', list(SH_TYPES), index=1)
        explain = st.checkbox('Generate SHAP violin and local explanation', value=True)
        nref = st.slider('Maximum training reference observations for SHAP', 25, 1000, 250, step=25)
    submitted = st.form_submit_button('Predict UCS', type='primary', use_container_width=True)

if submitted:
    row = pd.DataFrame([[ROCKS[rock], vp, pls, sh, SH_TYPES[shtype]]], columns=FEATURES)
    try:
        result = float(model.predict(preprocess(row, poly, scaler))[0])
        st.session_state['last_result'] = (result, row, rock, explain, nref)
    except Exception as exc:
        st.error(f'Prediction failed: {exc}')

if 'last_result' in st.session_state:
    result, row, selected_rock, explain, nref = st.session_state['last_result']
    st.subheader('Training-domain context')
    try:
        ref = reference()
        subset = ref.loc[ref['Category'] == ROCKS[selected_rock]].copy()
        st.write(f'Available training reference observations for **{selected_rock}**: **{len(subset):,}**.')
        if len(subset):
            d = local_domain_distance(row, subset)
            st.caption(f'Nearest reference observation distance (range-normalized VP, PLS and SH): {d:.3f}. This is descriptive only, not a validated applicability threshold.')
        if explain and len(subset) >= 2:
            with st.spinner('Computing category-specific SHAP values…'):
                fig = shap_chart(model, poly, scaler, row, subset, nref)
                st.pyplot(fig, clear_figure=False)
                buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=220, bbox_inches='tight')
                st.download_button('Download SHAP figure (PNG)', buf.getvalue(), file_name='UCS_SHAP.png', mime='image/png')
                plt.close(fig)
        elif explain:
            st.warning('Insufficient reference observations for a SHAP violin plot.')
    except Exception as exc:
        st.warning(f'Reference/SHAP analysis unavailable: {exc}. UCS prediction remains available.')
    st.divider()
    st.subheader('Prediction output')
    st.metric('Predicted UCS', f'{result:.2f} MPa')
    output = row.copy(); output['UCS_predicted (MPa)'] = result
    st.download_button('Download prediction (CSV)', output.to_csv(index=False).encode(),
                       file_name='UCS_prediction.csv', mime='text/csv')
    st.warning(DISCLAIMER)

st.divider()
st.caption('© 2026 IA Tijani, IA Lawal & collaborators. All rights reserved. | Research-use prediction tool. Verify suitability independently before engineering application.')
