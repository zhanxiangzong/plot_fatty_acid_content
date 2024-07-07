import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind

def plot_fatty_acid_content(fatty_acid_types, data_dict, use_name):
    group_names = ['WT'] + [name for name in data_dict if name != 'WT']
    num_groups = len(group_names)
    num_fatty_acids = len(fatty_acid_types)

    # Calculate means and standard deviations for plotting
    means = {group: [np.mean(data_dict[group][i]) for i in range(num_fatty_acids)] for group in group_names}
    stds = {group: [np.std(data_dict[group][i]) for i in range(num_fatty_acids)] for group in group_names}

    x = np.arange(num_fatty_acids)  # the label locations
    width = 1 / (num_groups + 1)  # the width of the bars

    sns.set(style="white")
    sns.set_palette("Set2")

    fig_width = 6 + num_groups  # dynamically adjust figure width
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    for i, group in enumerate(group_names):
        rects = ax.bar(x + (i - num_groups/2) * width, means[group], width, yerr=stds[group], label=group, capsize=5)

        # Perform t-tests and annotate significance for each fatty acid type
        if group != 'WT':
            p_values = [ttest_ind(data_dict['WT'][j], data_dict[group][j]).pvalue for j in range(num_fatty_acids)]
            for rect, p, std in zip(rects, p_values, stds[group]):
                height = rect.get_height()
                offset = std * 1.5  # Dynamic offset based on the error bar height
                if p < 0.001:
                    ax.annotate('***',
                                xy=(rect.get_x() + rect.get_width() / 2, height + offset),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                elif p < 0.01:
                    ax.annotate('**',
                                xy=(rect.get_x() + rect.get_width() / 2, height + offset),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                elif p < 0.05:
                    ax.annotate('*',
                                xy=(rect.get_x() + rect.get_width() / 2, height + offset),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('')
    ax.set_ylabel('Content(%)')

    ax.set_xticks(x)
    ax.set_xticklabels(fatty_acid_types, rotation=45, ha='right')
    ax.legend()
    ax.set_title(str(use_name))

    # Remove grid lines and keep only left and bottom spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    fig.tight_layout()
    st.pyplot(fig)

def convert_to_dict_and_categories(df, group_col):
    data_cols = [col for col in df.columns if col != group_col]
    result_dict = df.groupby(group_col).apply(lambda x: [x[col].tolist() for col in data_cols]).to_dict()
    return result_dict

# Streamlit app
st.title("脂肪酸含量分析")

uploaded_file = st.file_uploader("上传你的xlsx文件", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
    df['TYPE'] = [ ''.join(i.split(' ')[1:-1]) for i in df['Samples']]
    df['Group'] = [ ''.join(i.split(' ')[-1]) for i in df['Samples']]
    df['sub_group']=['_'.join(i.split('-')[:-1]) for i in df['Group']]
    df = df.loc[:, ~df.columns.isin(['Samples','Group',''])]
    st.write("数据预览：")
    st.dataframe(df)

    # Ensure only one TYPE is present
    if len(list(set(df['TYPE']))) != 1:
        st.error("一次只能分析一组数据，请确保输入数据仅包含一个 TYPE。")
    else:
        df = df.loc[:, ~df.columns.isin(['TYPE'])]
        group_col = st.selectbox("选择分组列", df.columns)
        
        if st.button("生成图表"):
            data_dict = convert_to_dict_and_categories(df, group_col)
            fatty_acid_types = [col for col in df.columns if col != group_col]
            use_name = st.text_input("请输入图表标题", "脂肪酸含量")
            plot_fatty_acid_content(fatty_acid_types, data_dict, use_name)
