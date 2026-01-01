import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import datetime
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

# ---------------------------------------------------------
# 0. Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="Ultimate Sci-Stat & Graph Engine", layout="wide")

# ---------------------------------------------------------
# 1. Sidebar Settings (Sidebar UI)
# ---------------------------------------------------------
with st.sidebar:
    # --- Top: Notice ---
    st.markdown("### 【Notice】")
    st.info("""
    This tool is a beta version. If you plan to use results from this tool in your publications or conference presentations, **please contact the developer (Seiji Kaneko) in advance.**

    👉 **[Contact & Feedback Form](https://forms.gle/xgNscMi3KFfWcuZ1A)**

    We will provide guidance on validation support and proper acknowledgments/co-authorship.
    """)
    st.divider()

    # --- Middle: Graph Settings ---
    st.header("🛠️ Graph Settings")
    
    with st.expander("📈 Chart Type", expanded=True):
        # Translated options
        graph_type = st.selectbox("Format", ["Bar Plot", "Box Plot", "Violin Plot"])
        
        # Logic updated to match English labels
        if "Bar" in graph_type:
            error_type = st.radio("Error Bars", ["SD (Standard Deviation)", "SEM (Standard Error of the Mean)"])
        else:
            error_type = "None"
        
    with st.expander("🎨 Design Tweaks", expanded=True):
        fig_title = st.text_input("Figure Title", value="Experiment Result")
        y_axis_label = st.text_input("Y-axis Label", value="Relative Value")
        manual_y_max = st.number_input("Y-axis Max (0 for Auto)", value=0.0, step=1.0)
        
        st.divider()
        st.caption("Plot Adjustments")
        bar_width = st.slider("Bar/Box Width", 0.1, 1.0, 0.6)
        dot_size = st.slider("Dot Size", 0, 100, 20)
        dot_alpha = st.slider("Dot Transparency", 0.1, 1.0, 0.7)
        jitter_strength = st.slider("Jitter", 0.0, 0.2, 0.04, 0.01)
        fig_height = st.slider("Figure Height", 3.0, 10.0, 5.0)

# ---------------------------------------------------------
# 2. Main Area: Data Input
# ---------------------------------------------------------
st.title("🔬 Ultimate Sci-Stat & Graph Engine")
st.markdown("""
**Integrated tool for automating statistical analysis to publication-quality graph creation (Pro Ver.)**
Automatically diagnoses data properties, selects optimal tests, and instantly creates graphs with significance bars.
""")

st.subheader("1. Data Input")
tab_manual, tab_csv = st.tabs(["✍️ Manual Input", "📂 CSV Upload"])

data_dict = {}

# --- A. Manual Input Mode ---
with tab_manual:
    if 'g_count' not in st.session_state: st.session_state.g_count = 3
    
    col_ctrl, _ = st.columns([1, 5])
    with col_ctrl:
        c1, c2 = st.columns(2)
        if c1.button("＋ Add"): st.session_state.g_count += 1
        if c2.button("－ Remove") and st.session_state.g_count > 2: st.session_state.g_count -= 1

    cols = st.columns(min(st.session_state.g_count, 4))
    for i in range(st.session_state.g_count):
        with cols[i % 4]:
            def_name = f"Group {i+1}"
            name = st.text_input(f"Name {i+1}", value=def_name, key=f"n{i}")
            raw = st.text_area(f"Data {i+1}", height=120, key=f"d{i}", placeholder="10.5\n12.3")
            vals = [float(x.strip()) for x in raw.replace(',', '\n').split('\n') if x.strip()]
            if len(vals) > 0: data_dict[name] = vals

# --- B. CSV Upload Mode ---
with tab_csv:
    uploaded_file = st.file_uploader("Upload CSV File (Columns: Group, Value)", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if len(df.columns) >= 2:
                g_col = df.columns[0]
                v_col = df.columns[1]
                for g_name in df[g_col].unique():
                    g_vals = df[df[g_col] == g_name][v_col].dropna().tolist()
                    if len(g_vals) > 0:
                        data_dict[g_name] = g_vals
                st.success(f"CSV Loaded Successfully: Detected {len(data_dict)} groups")
            else:
                st.error("CSV must have at least 2 columns (e.g., Column A: Group, Column B: Value)")
        except Exception as e:
            st.error(f"Load Error: {e}")

# ---------------------------------------------------------
# 3. Sidebar Addition: Group Color Settings
# ---------------------------------------------------------
group_colors = {}
if data_dict:
    with st.sidebar:
        with st.expander("🖍️ Group Color Settings", expanded=True):
            default_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]
            for i, g_name in enumerate(data_dict.keys()):
                col_def = default_colors[i % len(default_colors)]
                group_colors[g_name] = st.color_picker(f"{g_name} Color", col_def)

# ---------------------------------------------------------
# 4. Statistical Analysis Engine (Logic Core)
# ---------------------------------------------------------
def get_sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

sig_pairs = [] 

if len(data_dict) >= 2:
    st.header("2. Statistical Analysis Report")
    
    group_names = list(data_dict.keys())
    all_values = list(data_dict.values())
    
    # Diagnosis: Data Count Check
    valid_data_count = all(len(v) >= 2 for v in all_values)
    
    if not valid_data_count:
        st.warning("Please enter at least two numerical values for each group.")
    else:
        # Normality Diagnosis
        all_normal = True
        for v in all_values:
            if len(v) >= 3:
                _, p_s = stats.shapiro(v)
                if p_s <= 0.05: all_normal = False
        
        # Homogeneity of Variance Diagnosis
        try:
            _, p_lev = stats.levene(*all_values)
            is_equal_var = (p_lev > 0.05)
        except:
            is_equal_var = True

        # Test Logic
        method_name = ""
        p_global = 1.0
        
        # --- 2 Groups ---
        if len(data_dict) == 2:
            g1, g2 = all_values[0], all_values[1]
            if all_normal:
                method_name = "Student's t-test" if is_equal_var else "Welch's t-test"
                _, p_global = stats.ttest_ind(g1, g2, equal_var=is_equal_var)
            else:
                method_name = "Mann-Whitney U test"
                _, p_global = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                
            if p_global < 0.05:
                sig_pairs.append({'g1': group_names[0], 'g2': group_names[1], 'label': get_sig_label(p_global), 'p': p_global})

        # --- 3 or more Groups ---
        else:
            if all_normal and is_equal_var:
                method_name = "One-way ANOVA + Tukey's HSD"
                _, p_global = stats.f_oneway(*all_values)
                
                if p_global < 0.05:
                    flat_data = [v for sub in all_values for v in sub]
                    labels = [n for n, sub in data_dict.items() for _ in sub]
                    res = pairwise_tukeyhsd(flat_data, labels)
                    
                    df_res = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
                    for _, row in df_res.iterrows():
                        if row['reject']:
                            sig_pairs.append({'g1': row['group1'], 'g2': row['group2'], 'label': get_sig_label(row['p-adj']), 'p': row['p-adj']})
            else:
                method_name = "Kruskal-Wallis + Dunn's test"
                _, p_global = stats.kruskal(*all_values)
                
                if p_global < 0.05:
                    dunn = sp.posthoc_dunn(all_values, p_adjust='bonferroni')
                    dunn.columns = group_names
                    dunn.index = group_names
                    
                    for i in range(len(group_names)):
                        for j in range(i+1, len(group_names)):
                            n1, n2 = group_names[i], group_names[j]
                            p_val = dunn.loc[n1, n2]
                            if p_val < 0.05:
                                sig_pairs.append({'g1': n1, 'g2': n2, 'label': get_sig_label(p_val), 'p': p_val})

        # Generate Explanation (English)
        if all_normal and is_equal_var:
            easy_reason = "Data distribution was not skewed and variance was homogeneous, so the standard and high-power 'Parametric Test' was selected."
        elif not all_normal:
            easy_reason = "Data showed extreme skewness or outliers, so a 'Non-parametric Test' (robust against outliers and based on rank) was selected."
        else:
            easy_reason = "Significant difference in variance between groups was detected, so a method correcting for this difference was selected."

        result_summary = "【Significant Difference Found】 A clear, non-random difference was found between groups." if p_global < 0.05 else "【No Significant Difference】 The observed difference is likely within the range of error."

        # Display Report
        st.success(f"**Adopted Method: {method_name}**")
        
        with st.expander("📝 Ready-to-use Report (Detail)", expanded=True):
            full_report = f"""
【Analysis Report: Comparison of {", ".join(group_names)}】

1. Purpose of Analysis:
   Checked for statistically meaningful differences in the means of each group.

2. Method Selected (Reason):
   Method: {method_name}
   Reason: {easy_reason}
   * The most scientifically valid procedure was selected after checking data normality and homoscedasticity.

3. Results:
   Judgment: {result_summary}
   Global P-value: {p_global:.4e}
   (*If P < 0.05, it is judged as statistically significant*)

4. Post-hoc Comparisons:
   {"Since there are 3+ groups, all pairs were compared using strict criteria." if len(data_dict) > 2 else "Two groups were directly compared."}

5. Conclusion:
   Statistical evidence was obtained. A graph with significance labels was created based on this content.
            """
            st.text_area("Copy-Paste Report", value=full_report, height=350)

    st.divider()

# ---------------------------------------------------------
# 5. Graph Generation Engine (Visualization Core)
# ---------------------------------------------------------
if len(data_dict) >= 1:
    st.header("3. Graph Generation (Auto-Labeling)")
    
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        fig, ax = plt.subplots(figsize=(6, fig_height))
        
        group_names = list(data_dict.keys())
        x_positions = np.arange(len(group_names))
        
        # Calculate Y-axis max (Safety)
        all_vals_flat = [v for sub in data_dict.values() for v in sub if len(sub) > 0]
        max_val = np.max(all_vals_flat) if all_vals_flat else 1.0
        
        # --- A. Plotting ---
        for i, (name, vals) in enumerate(data_dict.items()):
            if len(vals) == 0: continue
            vals = np.array(vals)
            
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0
            sem_v = std_v / np.sqrt(len(vals)) if len(vals) > 0 else 0
            err = sem_v if error_type == "SEM (Standard Error of the Mean)" else std_v
            
            my_color = group_colors.get(name, "#333333")

            if "Bar" in graph_type:
                ax.bar(i, mean_v, width=bar_width, color=my_color, edgecolor='black', alpha=0.8, zorder=1)
                ax.errorbar(i, mean_v, yerr=err, fmt='none', color='black', capsize=5, zorder=2)
            elif "Box" in graph_type:
                ax.boxplot(vals, positions=[i], widths=bar_width, patch_artist=True,
                           boxprops=dict(facecolor=my_color, alpha=0.8), medianprops=dict(color='black'), showfliers=False)
            elif "Violin" in graph_type:
                parts = ax.violinplot(vals, positions=[i], widths=bar_width, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(my_color)
                    pc.set_alpha(0.8)
            
            if dot_size > 0:
                noise = np.random.normal(0, jitter_strength, len(vals))
                ax.scatter(x_positions[i] + noise, vals, s=dot_size, color='white', edgecolor='gray', zorder=3, alpha=dot_alpha)

        # --- B. Significance Bars ---
        y_step = max_val * 0.15
        current_y = max_val * 1.15 # Initial position slightly higher
        
        for pair in sig_pairs:
            try:
                idx1 = group_names.index(pair['g1'])
                idx2 = group_names.index(pair['g2'])
                x1, x2 = idx1, idx2
                bar_h = current_y
                col_h = max_val * 0.03
                
                ax.plot([x1, x1, x2, x2], [bar_h-col_h, bar_h, bar_h, bar_h-col_h], lw=1.5, c='black')
                ax.text((x1+x2)/2, bar_h, pair['label'], ha='center', va='bottom', fontsize=14)
                
                current_y += y_step
            except: pass

        # --- C. Layout ---
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_names, fontsize=12)
        ax.set_ylabel(y_axis_label, fontsize=12)
        ax.set_title(fig_title, fontsize=14)
        
        # Y-axis Range
        if manual_y_max > 0:
            ax.set_ylim(0, manual_y_max)
        else:
            ax.set_ylim(0, current_y * 1.1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
        
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button("📥 Save Image (PNG)", data=img_buf, file_name=f"result_{now_str}.png", mime="image/png")
        
    except Exception as e:
        st.error(f"Rendering Error: {e}")

else:
    st.info("Please input data (Manual or CSV)")

# ---------------------------------------------------------
# 6. Sidebar Bottom: Disclaimer
# ---------------------------------------------------------
with st.sidebar:
    st.divider()
    st.caption("【Disclaimer】")
    st.caption("""
    This tool is for assistive purposes. Final interpretations and conclusions 
    should be made by the user based on professional expertise.
    """)
