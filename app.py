from enum import Enum
import spekpy as spk
import numpy as np
import pandas as pd
import streamlit as st
import uncertainties as un
import altair as alt


def build_and_run():
    # Page and style config
    st.set_page_config(page_title="Sebastian's DE Sims",
                       page_icon="⚛",
                       layout="centered",
                       initial_sidebar_state="expanded")
    st.markdown("<style>.css-r2zp1w { width: 35px !important; }</style>", unsafe_allow_html=True)
    st.markdown("<style>#MainMenu, footer {visibility: hidden;}</style>", unsafe_allow_html=True)

    # Title and info
    st.title("Dual Energy Quality Metric Test")
    st.markdown("This is a first-order simulation of image quality metrics for X-ray Dual-Energy images.")
    st.info("Use the sidebar to modify simulation parameters")

    # Sidebar: Loads simulation parameters into sim dictionary
    mats = spk.IO.get_matls()[1]
    sim = {}

    st.sidebar.subheader("Exposure Settings")
    col1, col2 = st.sidebar.beta_columns(2)
    sim['exp'] = [{}, {}]
    sim['exp'][0]['kVp'] = col1.number_input(label="Low kVp", value=60, min_value=31, max_value=80, step=5)
    sim['exp'][1]['kVp'] = col2.number_input(label="High kVp", value=120, min_value=81, max_value=150, step=5)
    sim['exp'][0]['mAs'] = col1.number_input(label="Low mAs", value=2.5, min_value=0.5, max_value=100., step=0.1, format="%.1f")
    sim['exp'][1]['mAs'] = col2.number_input(label="High mAs", value=2.5, min_value=0.5, max_value=100., step=0.1, format="%.1f")
    sim['SID'] = st.sidebar.slider(label="Source to Imager Distance (cm)", value=180, min_value=10, max_value=300, step=5)

    st.sidebar.subheader("Detector Details")
    sim['det'] = {}
    sim['det']['mat'] = st.sidebar.selectbox(label="Scintillator Material", options=mats, index=mats.index("Cesium Iodide"))
    col1, col2 = st.sidebar.beta_columns(2)
    sim['det']['t'] = col1.number_input(label="Scint. Thickness (μm)", value=500, min_value=100, max_value=2000, step=50)
    sim['det']['a'] = col2.number_input(label="Pixel Aperature (μm)", value=100, min_value=10, max_value=500, step=10)

    st.sidebar.subheader("Phantom Base")
    sim['base'] = [{}, {}]
    sim['base'][0]['mat'] = st.sidebar.selectbox(label="Base Material 1", options=mats, index=mats.index("Polymethyl Methacrylate (Lucite Perspex or Plexiglas)"))
    sim['base'][1]['mat'] = st.sidebar.selectbox(label="Base Material 2", options=mats, index=mats.index("Aluminum Alloy (Type 6061)"))
    col1, col2 = st.sidebar.beta_columns(2)
    sim['base'][0]['t'] = col1.number_input(label="Base 1 Thickness (mm)", value=73., min_value=0., max_value=500., step=0.1, format="%.1f")
    sim['base'][1]['t'] = col2.number_input(label="Base 2 Thickness (mm)", value=4.1, min_value=0., max_value=500., step=0.1, format="%.1f")

    st.sidebar.subheader("Phantom Inserts")
    sim['inserts'] = {'soft': {}, 'hard': {}}

    sim['inserts']['soft']['mat'] = st.sidebar.selectbox(label="Soft Insert Material", options=mats, index=mats.index("Polymethyl Methacrylate (Lucite Perspex or Plexiglas)"))
    col1, col2 = st.sidebar.beta_columns(2)
    sim['inserts']['soft']['step'] = col1.number_input(label="Soft Step (mm)", value=2.0, min_value=0.5, max_value=10., step=0.5, format="%.1f")
    sim['inserts']['soft']['count'] = col2.number_input(label="Soft Features Count", value=5, min_value=2, max_value=11)

    sim['inserts']['hard']['mat'] = st.sidebar.selectbox(label="Hard Insert Material", options=mats, index=mats.index("Aluminum Alloy (Type 6061)"))
    col1, col2 = st.sidebar.beta_columns(2)
    sim['inserts']['hard']['step'] = col1.number_input(label="Hard Step (mm)", value=0.5, min_value=0.5, max_value=10., step=0.5, format="%.1f")
    sim['inserts']['hard']['count'] = col2.number_input(label="Hard Features Count", value=5, min_value=2, max_value=11)

    # Calculations
    # 1) Get pixel signal and variance for all exposures and phantom inserts
    signals = []
    for exp in sim['exp']:
        # Create input spectrum
        spec = spk.Spek(kvp=exp['kVp'], z=sim['SID'], mas=exp['mAs'])
        spec.filter('Al', 1.6)  # fixed intrinsic filtration

        # Phantom filtration
        for base in sim['base']:
            spec.filter(base['mat'], base['t'])

        # Insert filtration
        signal = {}
        for tissue in sim['inserts'].keys():
            signal[tissue] = np.array(
                [calc_detector_signal(
                    spk.Spek.clone(spec).filter(sim['inserts'][tissue]['mat'], i*sim['inserts'][tissue]['step']),
                    sim['det']['mat'], sim['det']['t'], sim['det']['a'])
                 for i in range(sim['inserts'][tissue]['count']+1)]
            )
        signals.append(signal)

    # 2) Get the Dual-Energy signals and store them in data frames
    dfs = {}
    for tissue in sim['inserts'].keys():
        central = (sim['inserts'][tissue]['count']+1) // 2
        w = np.log(signals[-1][tissue][0].n/signals[-1][tissue][central].n) / \
            np.log(signals[+0][tissue][0].n/signals[+0][tissue][central].n)
        for img in sim['inserts'].keys():
            de = signals[-1][img] / signals[0][img] ** w
            dfs[(img, tissue)] = pd.DataFrame.from_dict({
                'feature': range(len(de)),
                'thickness': [i * sim['inserts'][tissue]['step'] for i in range(len(de))],
                'signal': [v.n for v in de],
                'noise': [v.s for v in de],
                'contrast': [v.n - de[0].n for v in de],
                'cnr': [(v.n - de[0].n) / np.sqrt(v.s**2 + de[0].s**2) for v in de],
            })

    # Show Raw Data
    with st.beta_expander("Show Simulation Data..."):
        for comb in dfs.keys():
            st.subheader("{1} features in {0}-subtracted image".format(*comb))
            dfs[comb]
    st.text("")  #spacing

    # Plots
    for plot in ['contrast', 'cnr']:
        cs = []
        for img in sim['inserts'].keys():
            c1 = alt.Chart(dfs[(img, 'soft')]).mark_line(color='#5276A7').encode(
                alt.X('feature', title='Feature #', ),
                alt.Y(plot, title='Soft Feature Contrast', axis=alt.Axis(titleColor='#5276A7'))
            )
            c2 = alt.Chart(dfs[(img, 'hard')]).mark_line(color='#57A44C').encode(
                alt.X('feature', title='Feature #'),
                alt.Y(plot, title='Hard Feature Contrast', axis=alt.Axis(titleColor='#57A44C'))
            )
            c = alt.layer(c1, c2).resolve_scale(y='independent').properties(
                title="%s in %s-tissue-subtracted image" % (plot, img),
                width=500
            )
            cs.append(c)
        st.altair_chart(cs[0] & cs[1])

    # Info and Copyright
    st.markdown("Find source code on [github](https://github.com/Sebolains/de-sim-dash/)")
    st.text("© 2021, Sebastian Maurino")


def calc_detector_signal(spectrum: spk.Spek, scint: str, t: float, a: float) -> un.ufloat:
    """
    Calculates the detector pixel signal and variance given a spectrum and scintillator properties
    
    :param spectrum: spekpy Spek object containing input spectrum
    :param scint: Name of scintillator material
    :param t: Thickness of the scintillator [mm]
    :param a: Pixel aperature (or length of square pixel side) [μm]
    :return: ufloat absorbed energy in pixel with spatial variance
    """
    k, s0 = spectrum.get_spectrum()
    s1 = spk.Spek.clone(spectrum).filter(scint, t*1e-3).get_spk()
    assert(s0.shape == s1.shape)
    s = s0 - s1  #[photons cm⁻² keV⁻¹]
    e = (a * 1e-4)**2 * np.trapz(x=k, y=k * s)  #energy deposited
    σ = (a * 1e-4) * np.sqrt(np.trapz(x=k, y=k**2 * s))  #poissonian variance

    return un.ufloat(e, σ)


if __name__ == '__main__':
    build_and_run()