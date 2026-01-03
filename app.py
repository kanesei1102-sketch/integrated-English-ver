import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import datetime
from scipy import stats
try:
    import scikit_posthocs as sp
except ImportError:
    sp = None
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------------------------------------------------------
# 0. Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="Ultimate Sci-Stat & Graph Engine", layout="wide")

# ---------------------------------------------------------
# 1. Sidebar Settings
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 【Notice】")
    st.info("""
    This tool is a beta version. If you plan to use results from this tool in your publications or conference presentations, **please contact the developer (Seiji Kaneko) in advance.**

    👉 **[Contact & Feedback Form](https://forms.gle/xgNscMi3KFfWcuZ1A)**

    We will provide guidance on validation support and proper acknowledgments/co-authorship.
    """)
    st.divider()

    st.header("🛠️ Graph Configuration")
    
    with st.expander("📈 Plot Type", expanded=True):
        graph_type = st.selectbox("Style", ["Bar Plot", "Box Plot", "Violin Plot"])
        if "Bar" in graph_type:
            error_type = st.radio("Error Bar", ["SD (Standard Deviation)", "SEM (Standard Error)"])
        else:
            error_type = "None"
        
    with st.expander("🎨 Layout & Appearance", expanded=True):
        fig_title = st.text_input("Figure Title", value="Experiment Result")
        y_axis_label = st.text_input("Y-axis Label", value="Relative Value")
        manual_y_max = st.number_input("Y-axis Max (0 for Auto)", value=0.0, step=1.0)
        
        st.divider()
        st.caption("Spacing & Width")
        
        # English Labels for Sliders
        group_spacing = st.slider("↔️ Group Spacing (X-axis)", 0.8, 3.0, 1.2, 0.1, help="Adjust the distance between groups.")
        bar_width = st.slider("⬛ Box/Bar Width", 0.1, 1.5, 0.6, 0.1, help="Adjust the width of the bars/boxes.")
        
        st.caption("Dots & Dimensions")
        dot_size = st.slider("Dot Size", 0, 100, 20)
        dot_alpha = st.slider("Dot Opacity", 0.1, 1.0, 0.7)
        jitter_strength = st.slider("Jitter Strength", 0.0, 0.2, 0.04, 0.01)
        fig_height = st.slider("Figure Height", 3.0, 10.0, 5.0)

# ---------------------------------------------------------
# 2. Main Area: Data Input
# ---------------------------------------------------------
st.title("🔬 Ultimate Sci-Stat & Graph Engine")
st.markdown("""
**Automated Statistical Analysis & Scientific Graphing Tool (Pro Ver.)** Automatically diagnoses data properties (Normality/Homogeneity), selects the optimal statistical test, and generates publication-ready graphs with significance bars.
""")

st.subheader("1. Data Input")
tab_manual, tab_csv = st.tabs(["✍️ Manual Input", "📂 CSV Upload"])

data_dict = {}

# --- Manual Input ---
with tab_manual:
    if 'g_count' not in st.session_state: st.session_state.g_count = 3
    col_ctrl, _ = st.columns([1, 5])
    with col_ctrl:
        c1, c2 = st.columns(2)
        if c1.button("＋ Add"): st.session_state.g_count += 1
        if c2.button("－ Del") and st.session_state.g_count > 2: st.session_state.g_count -= 1

    cols = st.columns(min(st.session_state.g_count, 4))
    for i in range(st.session_state.g_count):
        with cols[i % 4]:
            def_name = f"Group {i+1}"
            name = st.text_input(f"Name {i+1}", value=def_name, key=f"n{i}")
            raw = st.text_area(f"Data {i+1}", height=120, key=f"d{i}", placeholder="10.5\n12.3")
            vals = [float(x.strip()) for x in raw.replace(',', '\n').split('\n') if x.strip()]
            if len(vals) > 0: data_dict[name] = vals

# --- CSV Upload ---
with tab_csv:
    uploaded_file = st.file_uploader("Upload CSV (Columns: Group, Value)", type="csv")
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
                st.success(f"CSV Loaded Successfully: {len(data_dict)} groups detected")
            else:
                st.error("CSV must have at least 2 columns (e.g., Col A = Group Name, Col B = Values)")
        except Exception as e:
            st.error(f"Load Error: {e}")

# ---------------------------------------------------------
# 3. Group Color Settings
# ---------------------------------------------------------
group_colors = {}
if data_dict:
    with st.sidebar:
        with st.expander("🖍️ Color Settings", expanded=True):
            default_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]
            for i, g_name in enumerate(data_dict.keys()):
                col_def = default_colors[i % len(default_colors)]
                group_colors[g_name] = st.color_picker(f"{g_name} Color", col_def)

# ---------------------------------------------------------
# 4. Statistical Analysis Engine
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
    valid_data_count = all(len(v) >= 2 for v in all_values)
    
    if not valid_data_count:
        st.warning("Please input at least 2 numeric values per group.")
    else:
        # Normality & Homogeneity
        all_normal = True
        for v in all_values:
            if len(v) >= 3:
                _, p_s = stats.shapiro(v)
                if p_s <= 0.05: all_normal = False
        try:
            _, p_lev = stats.levene(*all_values)
            is_equal_var = (p_lev > 0.05)
        except:
            is_equal_var = True

        # Test Selection Logic
        method_name = ""
        p_global = 1.0
        
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
                if p_global < 0.05 and sp is not None:
                    dunn = sp.posthoc_dunn(all_values, p_adjust='bonferroni')
                    dunn.columns = group_names
                    dunn.index = group_names
                    for i in range(len(group_names)):
                        for j in range(i+1, len(group_names)):
                            n1, n2 = group_names[i], group_names[j]
                            if dunn.loc[n1, n2] < 0.05:
                                sig_pairs.append({'g1': n1, 'g2': n2, 'label': get_sig_label(dunn.loc[n1, n2]), 'p': dunn.loc[n1, n2]})

        # Report Text Generation
        if all_normal and is_equal_var:
            easy_reason = "Data distribution is normal and variances are equal. Selected the standard 'Parametric Test' for highest statistical power."
        elif not all_normal:
            easy_reason = "Data distribution is non-normal or contains outliers. Selected 'Non-Parametric Test' (Rank-based) for robustness."
        else:
            easy_reason = "Variances between groups are unequal. Selected a test with appropriate corrections (e.g., Welch's t-test)."

        result_summary = "【Significant】 A statistically significant difference was found between groups." if p_global < 0.05 else "【Not Significant】 No statistically significant difference was found."

        analysis_path = f"""
【Automated Diagnosis Process】
1. Normality Check (Shapiro-Wilk): {"PASS (Normal)" if all_normal else "FAIL (Non-Normal)"}
2. Homogeneity Check (Levene): {"PASS (Equal Var)" if is_equal_var else "FAIL (Unequal Var)"}
⇒ Based on the above, the algorithm automatically selected: "{method_name}".
"""
        st.success(f"**Selected Method: {method_name}**")
        with st.expander("📝 Formal Statistical Report (Copy & Paste)", expanded=True):
            full_report = f"""
【Statistical Analysis Report: Comparison of {", ".join(group_names)}】

{analysis_path}

1. Objective:
   To determine if there is a statistically significant difference between the means of the groups.

2. Methodology & Rationale:
   Selected Test: {method_name}
   Reason: {easy_reason}
   (The test was chosen based on automated pre-checks for Normality and Homogeneity of variance.)

3. Results:
   Conclusion: {result_summary}
   Global P-value: {p_global:.4e}
   (P < 0.05 indicates statistical significance.)

4. Post-hoc Analysis (Multiple Comparisons):
   {"Pairwise comparisons were conducted to identify specific group differences." if len(data_dict) > 2 else "Direct comparison between two groups was conducted."}

5. Summary:
   Based on the analysis, statistical evidence was obtained from the data. Significance labels have been automatically applied to the generated graph.
            """
            st.text_area("Report Output", value=full_report, height=450)
    st.divider()

# ---------------------------------------------------------
# 5. Graph Generation Engine
# ---------------------------------------------------------
if len(data_dict) >= 1:
    st.header("3. Graph Generation (Auto-Labeling)")
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        
        # --- Layout Calculation ---
        base_scale = 1.5
        auto_width = max(6.0, len(data_dict) * base_scale * group_spacing)
        
        fig, ax = plt.subplots(figsize=(auto_width, fig_height))
        
        group_names = list(data_dict.keys())
        
        # X positions based on spacing
        x_positions = np.arange(len(group_names)) * group_spacing
        
        all_vals_flat = [v for sub in data_dict.values() for v in sub if len(sub) > 0]
        max_val = np.max(all_vals_flat) if all_vals_flat else 1.0
        
        # --- A. Plot Drawing ---
        for i, (name, vals) in enumerate(data_dict.items()):
            if len(vals) == 0: continue
            vals = np.array(vals)
            pos = x_positions[i]
            
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0
            sem_v = std_v / np.sqrt(len(vals)) if len(vals) > 0 else 0
            err = sem_v if error_type == "SEM" else std_v
            my_color = group_colors.get(name, "#333333")

            if "Bar" in graph_type:
                ax.bar(pos, mean_v, width=bar_width, color=my_color, edgecolor='black', alpha=0.8, zorder=1)
                ax.errorbar(pos, mean_v, yerr=err, fmt='none', color='black', capsize=5, zorder=2)
            elif "Box" in graph_type:
                ax.boxplot(vals, positions=[pos], widths=bar_width, patch_artist=True,
                           boxprops=dict(facecolor=my_color, alpha=0.8), medianprops=dict(color='black'), showfliers=False)
            elif "Violin" in graph_type:
                parts = ax.violinplot(vals, positions=[pos], widths=bar_width, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(my_color); pc.set_alpha(0.8)
            
            if dot_size > 0:
                noise = np.random.normal(0, jitter_strength, len(vals))
                ax.scatter(pos + noise, vals, s=dot_size, color='white', edgecolor='gray', zorder=3, alpha=dot_alpha)

        # --- B. Significance Bars ---
        y_step = max_val * 0.15
        current_y = max_val * 1.15
        
        for pair in sig_pairs:
            try:
                idx1 = group_names.index(pair['g1'])
                idx2 = group_names.index(pair['g2'])
                x1, x2 = x_positions[idx1], x_positions[idx2]
                
                bar_h = current_y
                col_h = max_val * 0.03
                ax.plot([x1, x1, x2, x2], [bar_h-col_h, bar_h, bar_h, bar_h-col_h], lw=1.5, c='black')
                ax.text((x1+x2)/2, bar_h, pair['label'], ha='center', va='bottom', fontsize=14)
                current_y += y_step
            except: pass

        # --- C. Axis Settings ---
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_names, fontsize=12)
        ax.set_ylabel(y_axis_label, fontsize=12)
        ax.set_title(fig_title, fontsize=14)
        
        margin = 0.8 * group_spacing
        ax.set_xlim(min(x_positions) - margin, max(x_positions) + margin)

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
        st.download_button("📥 Download Image (PNG)", data=img_buf, file_name=f"result_{now_str}.png", mime="image/png")
    except Exception as e:
        st.error(f"Plot Error: {e}")
else:
    st.info("Please input data (Manual or CSV)")

with st.sidebar:
    st.divider()
    st.caption("【Disclaimer】")
    st.caption("""
    This tool is for assistive purposes. Final interpretations and conclusions 
    should be made by the user based on professional expertise.
    """)
