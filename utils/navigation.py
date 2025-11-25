import streamlit as st

# Configuration of groups and which pages belong to each group.
# Make sure the file paths below match your actual file names.
GROUP_CONFIG = {
    "Explorative / overview": [
        ("Home.py", "🏠 Home"),
        ("pages/2_Production_Elhub.py", "Elhub overview"),
        ("pages/4_Data_overview.py", "Data overview"),
        ("pages/5_Weather_Plots.py", "Weather plots"),
        ("pages/10_Meteo_Energy_Correlation.py", "Meteo–Energy"),
    ],
    "Anomalies / data quality": [
        ("Home.py", "🏠 Home"),
        ("pages/3_STL_and_Spectrogram.py", "STL & spectrogram"),
        ("pages/6_Outliers_and_LOF.py", "Outliers & LOF"),
    ],
    "Forecasting": [
        ("Home.py", "🏠 Home"),
        ("pages/11_Forecasting_SARIMAX.py", "Energy forecast (SARIMAX)"),
    ],
    "Snow & geo": [
        ("Home.py", "🏠 Home"),
        ("pages/8_Map_price_areas.py", "Price area map"),
        ("pages/9_Snow_drift.py", "Snow drift"),
    ],
}


def sidebar_navigation(active_group: str) -> str:
    """
    Render navigation in the sidebar:

    - Show all groups as visible items ("Groups")
    - Highlight the currently selected group
    - Show page links for the selected group

    Parameters
    ----------
    active_group : str
        The group that this page logically belongs to
        (e.g., "Explorative / overview").

    Returns
    -------
    str
        The group that is currently selected (could be useful
        if you want to use it elsewhere in the page).
    """

    # Determine which group should be active:
    # - Prefer the one stored in session_state (if any)
    # - Fallback to the group passed in by the page (active_group)
    selected_group = st.session_state.get("selected_group", active_group)
    if selected_group not in GROUP_CONFIG:
        selected_group = active_group

    st.sidebar.markdown("## Groups")

    # Render all groups as clickable items
    for group_name in GROUP_CONFIG.keys():
        key = f"group_btn_{group_name}"

        if group_name == selected_group:
            # Active group: display as highlighted text (no button)
            st.sidebar.markdown(f"**➤ {group_name}**")
        else:
            # Other groups: display as buttons
            if st.sidebar.button(group_name, key=key):
                st.session_state["selected_group"] = group_name
                selected_group = group_name

    st.sidebar.markdown("---")

    # Render the pages (page_link) for the selected group
    pages = GROUP_CONFIG.get(selected_group, [])
    for file_path, label in pages:
        st.sidebar.page_link(file_path, label=label)

    st.sidebar.markdown("---")

    return selected_group
