# 导入所需库
import streamlit as st
import requests
from bs4 import BeautifulSoup
import jieba
from collections import Counter
import re
from pyecharts import options as opts
from pyecharts.charts import (WordCloud, Bar, Line, Pie, Radar, Scatter, HeatMap, TreeMap)
from streamlit_echarts import st_pyecharts
import numpy as np

# 页面基础配置（宽屏显示，优化视觉效果）
st.set_page_config(
    page_title="URL文本词频分析系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- 侧边栏配置（满足作业要求：图表筛选+低频词过滤） ----------------------
st.sidebar.title("📊 功能筛选面板")

# 1. 图表类型筛选（下拉框，包含8种图表，满足至少7种要求）
st.sidebar.subheader("选择可视化图表")
chart_type = st.sidebar.selectbox(
    label="请选择要展示的图表类型",
    options=[
        "词云图",
        "词频排名柱状图",
        "词频趋势折线图",
        "词频占比饼图",
        "词频对比雷达图",
        "词频分布散点图",
        "词频热力图",
        "词频层级树状图"
    ],
    index=0  # 默认选中词云图
)

# 2. 低频词过滤（交互式滑块，满足作业交互要求）
st.sidebar.subheader("低频词过滤设置")
min_frequency = st.sidebar.slider(
    label="最低词频阈值（过滤低于该值的词汇）",
    min_value=1,
    max_value=20,
    value=2,
    step=1,
    help="滑动调整阈值，低于该值的词汇将被过滤，不参与统计和可视化"
)

# ---------------------- 主页面内容 ----------------------
st.title("🔍 URL文章词频分析与可视化平台")
st.divider()  # 分隔线，优化排版

# 1. URL输入框（满足作业：用户输入文章URL）
url = st.text_input(
    label="请输入文章URL地址",
    placeholder="示例：https://www.xxx.com/article.html",
    help="请确保URL可正常访问，优先选择无反爬限制的纯文本文章页面"
)

# 定义基础中文停用词表（优化分词效果，提升统计准确性）
STOP_WORDS = set([
    "的", "地", "得", "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "是", "在", "有", "就", "不", "和", "也", "都", "这", "那", "着", "了",
    "过", "将", "要", "能", "会", "可以", "对", "对于", "关于", "与", "及",
    "或", "一个", "一些", "这种", "那种", "这里", "那里", "什么", "怎么",
    "为什么", "哪", "哪一个", "谁", "如何", "哦", "啊", "呀", "呢", "吧",
    "吗", "嗯", "哈", "嘿", "喂", "哎", "呃", "且", "而", "若", "因", "为",
    "之", "其", "所", "以", "并", "还", "只", "又", "更", "最", "很", "挺"
])

# 2. 定义URL文本抓取函数（满足作业：请求URL抓取文本内容）
def fetch_url_text(url):
    """抓取URL对应的文章文本内容，处理异常并返回纯文本"""
    try:
        # 设置请求头，模拟浏览器访问，避免被反爬
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # 发送GET请求，设置超时时间
        response = requests.get(url, headers=headers, timeout=15)
        # 自动识别编码，避免中文乱码
        response.encoding = response.apparent_encoding
        # 解析HTML页面
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 过滤无效标签，提取正文
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        
        # 提取纯文本并清理空白字符
        raw_text = soup.get_text()
        clean_text = re.sub(r"\s+", " ", raw_text).strip()
        return clean_text
    
    except requests.exceptions.Timeout:
        st.error("❌ 请求超时！请检查URL是否可访问，或网络是否正常")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ 连接失败！请检查URL是否正确，或目标网站是否可访问")
        return None
    except Exception as e:
        st.error(f"❌ 抓取文本失败：{str(e)}")
        return None

# 3. 定义分词与词频统计函数（满足作业：对文本分词，统计词频）
def segment_and_count(text):
    """对文本进行jieba分词，过滤无效词汇，返回词频统计结果"""
    # 中文分词
    word_list = jieba.lcut(text)
    # 过滤条件：非停用词、长度>1、纯中文字符
    valid_words = [
        word for word in word_list
        if word not in STOP_WORDS
        and len(word) > 1
        and re.match(r"^[\u4e00-\u9fa5]+$", word)  # 确保是纯中文
    ]
    # 统计词频
    word_frequency = Counter(valid_words)
    return word_frequency

# 4. 定义图表创建函数（修复所有参数命名错误，符合pyecharts规范）
def generate_chart(chart_type, filtered_word_data, top20_words, top20_counts):
    """根据选择的图表类型，生成对应的pyecharts图表"""
    if chart_type == "词云图":
        # 词云图（满足作业词云要求）
        wordcloud = (
            WordCloud()
            .add(
                series_name="词汇词频",
                data_pair=filtered_word_data,
                word_size_range=[15, 60],  # 词汇大小范围
                shape="circle"  # 词云形状为圆形
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="文章词汇词云图",
                    subtitle="词越大表示词频越高",
                    title_textstyle_opts=opts.TextStyleOpts(font_size=20)
                ),
                tooltip_opts=opts.TooltipOpts(trigger="item", formatter="词汇：{b}<br/>词频：{c}")
            )
        )
        return wordcloud
    
    elif chart_type == "词频排名柱状图":
        # 横向柱状图，更易查看长词汇
        bar = (
            Bar()
            .add_xaxis(top20_words)
            .add_yaxis("词频数量", top20_counts, color="#1890ff")
            .reversal_axis()  # 横向翻转
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频排名前20柱状图", subtitle="横向展示更清晰"),
                xaxis_opts=opts.AxisOpts(name="词频"),
                yaxis_opts=opts.AxisOpts(name="词汇"),
                tooltip_opts=opts.TooltipOpts(formatter="{b}：{c}次")
            )
        )
        return bar
    
    elif chart_type == "词频趋势折线图":
        # 关键修复1：mark_point_opts -> markpoint_opts
        # 关键修复2：line_style_opts -> linestyle_opts（无大写S，全小写+下划线）
        line = (
            Line()
            .add_xaxis(top20_words)
            .add_yaxis(
                "词频趋势",
                top20_counts,
                markpoint_opts=opts.MarkPointOpts(
                    data=[opts.MarkPointItem(type_="max"), opts.MarkPointItem(type_="min")]
                ),
                linestyle_opts=opts.LineStyleOpts(width=3, color="#ff4d4f")  # 修复参数名
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频排名前20折线图", subtitle="展示词频变化趋势"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
                yaxis_opts=opts.AxisOpts(name="词频数量"),
                tooltip_opts=opts.TooltipOpts(trigger="item")
            )
        )
        return line
    
    elif chart_type == "词频占比饼图":
        # 修复：移除无效width参数，规范图例配置
        pie = (
            Pie()
            .add(
                "",
                list(zip(top20_words, top20_counts)),
                radius=["30%", "75%"],
                rosetype="radius"  # 玫瑰图样式，更美观
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频排名前20饼图", subtitle="展示各词汇词频占比"),
                legend_opts=opts.LegendOpts(orient="vertical", pos_left="10%")
            )
            .set_series_opts(
                tooltip_opts=opts.TooltipOpts(formatter="{b}：{c}次（{d}%）")
            )
        )
        return pie
    
    elif chart_type == "词频对比雷达图":
        # 雷达图取前8个词汇，避免过于拥挤
        top8_words = top20_words[:8]
        top8_counts = top20_counts[:8]
        radar = (
            Radar()
            .add_schema(
                schema=[opts.RadarIndicatorItem(name=word, max_=max(top8_counts)) for word in top8_words],
                shape="polygon"
            )
            .add("词频数据", [top8_counts], color="#52c41a")
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频前8雷达图", subtitle="多维度词汇词频对比"),
                legend_opts=opts.LegendOpts(selected_mode="single")
            )
        )
        return radar
    
    elif chart_type == "词频分布散点图":
        scatter = (
            Scatter()
            .add_xaxis(top20_words)
            .add_yaxis(
                "词频分布",
                top20_counts,
                symbol_size=12,
                itemstyle_opts=opts.ItemStyleOpts(color="#fa8c16")
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频排名前20散点图", subtitle="展示词汇与词频对应关系"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
                yaxis_opts=opts.AxisOpts(name="词频数量"),
                tooltip_opts=opts.TooltipOpts(formatter="{b}：{c}次")
            )
        )
        return scatter
    
    elif chart_type == "词频热力图":
        # 构造热力图二维数据，取前10个词汇
        top10_words = top20_words[:10]
        top10_counts = top20_counts[:10]
        heat_data = []
        for i in range(5):
            for j in range(2):
                idx = i * 2 + j
                heat_data.append([i, j, top10_counts[idx]])
        
        heatmap = (
            HeatMap()
            .add_xaxis([f"第{i+1}行" for i in range(5)])
            .add_yaxis("词汇", [top10_words[j] for j in range(0, 10, 2)], heat_data)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频前10热力图", subtitle="颜色越深词频越高"),
                visualmap_opts=opts.VisualMapOpts(
                    min_=min(top10_counts),
                    max_=max(top10_counts),
                    orient="horizontal",
                    pos_bottom="5%"
                ),
                tooltip_opts=opts.TooltipOpts(formatter="行{x}：{y} = {value}次")
            )
        )
        return heatmap
    
    elif chart_type == "词频层级树状图":
        # 构造树状图数据结构
        treemap_data = [
            {
                "name": "词频总览",
                "children": [{"name": word, "value": count} for word, count in zip(top20_words, top20_counts)]
            }
        ]
        treemap = (
            TreeMap()
            .add(
                "词频树状图",
                treemap_data,
                levels=[
                    opts.TreeMapLevelsOpts(
                        treemap_itemstyle_opts=opts.TreeMapItemStyleOpts(
                            border_color="#ffffff",
                            border_width=2,
                            gap_width=1
                        )
                    )
                ]
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频排名前20树状图", subtitle="层级化展示词汇词频"),
                tooltip_opts=opts.TooltipOpts(formatter="{b}：{c}次")
            )
        )
        return treemap

# ---------------------- 主逻辑执行流程 ----------------------
if url:
    # 1. 抓取URL文本
    with st.spinner("🔄 正在抓取文章内容，请稍候..."):
        article_text = fetch_url_text(url)
    
    if article_text:
        st.success("✅ 文章内容抓取成功！")
        
        # 展示文本预览（折叠面板，优化页面布局）
        with st.expander("📄 查看文章文本预览（前600字）", expanded=False):
            preview_text = article_text[:600] + "..." if len(article_text) > 600 else article_text
            st.text_area("文本预览", preview_text, height=150, disabled=True)
        
        # 2. 分词与词频统计
        with st.spinner("🔤 正在进行分词和词频统计，请稍候..."):
            word_freq_result = segment_and_count(article_text)
        
        # 3. 过滤低频词
        filtered_word_freq = {
            word: count for word, count in word_freq_result.items()
            if count >= min_frequency
        }
        
        # 处理过滤后无数据的情况
        if not filtered_word_freq:
            st.warning("⚠️ 过滤后无有效词汇，请降低左侧的词频阈值重试！")
        else:
            # 4. 排序并获取前20词汇（满足作业：展示词频排名前20）
            sorted_word_freq = sorted(filtered_word_freq.items(), key=lambda x: x[1], reverse=True)
            top20_word_freq = sorted_word_freq[:20]
            top20_words = [item[0] for item in top20_word_freq]
            top20_counts = [item[1] for item in top20_word_freq]
            
            # 展示前20词频表格（不依赖pyarrow，避免DLL错误）
            st.subheader("🏆 词频排名前20词汇")
            top20_df = {
                "排名": list(range(1, 21)),
                "词汇": top20_words,
                "词频": top20_counts
            }
            st.table(top20_df)  # 用table替代dataframe，无依赖问题
            
            # 5. 生成并展示图表
            st.subheader(f"📈 {chart_type}展示")
            with st.spinner("🎨 正在生成图表，请稍候..."):
                chart = generate_chart(chart_type, sorted_word_freq, top20_words, top20_counts)
                st_pyecharts(chart, height="600px")
else:
    # 未输入URL时的提示
    st.info("💡 请在上方输入框中填写有效的文章URL，即可开始词频分析之旅～")
    st.divider()
    st.caption("提示：该工具支持绝大多数中文文章页面，优先选择无反爬限制的纯文本文章URL")