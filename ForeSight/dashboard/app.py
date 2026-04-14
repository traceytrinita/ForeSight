import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

#finding utils folder inside dashboard
sys.path.append(str(Path(__file__).parent))

from utils.ai_insights import generate_ai_insight
    
#project root path
#parents[1]: ForeSight

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#building the path
#project_root/data/processed
DATA_DIR = PROJECT_ROOT / "data" / "processed"
    
#page config
st.set_page_config(
    page_title="ForeSight AI Dashboard",
    layout="wide"
)

st.markdown("""
<style>
:root{
    --navy-900:#0B1220;
    --navy-800:#0E1A33;
    --navy-700:#132B55;
    --blue-600:#2563EB;
    --blue-500:#3B82F6;
    --blue-100:#DBEAFE;
    --ink:#0B1220;
    --muted:#50607A;
    --card:#FFFFFF;
    --border:rgba(11,18,32,.10);
    --shadow:0 10px 28px rgba(11, 18, 2,.10);
}

.stApp {
    background: linear-gradient(180deg, #FFFFFF 0%, #F6F9FF 100%);
    color: var(--ink);
}

.block-container {padding-top: 1.0rem; padding-bottom: 2.2rem; }

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(19, 43, 85, .08), rgba(255,255,255,1));
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] *{color: var(--ink);}

/* Hero header */
.fs-hero{
    padding: 16px 18px;
    border-radius: 19px;
    border: 1px solid var(--border);
    background:
        radial-gradient(900px 200px at 10% 0%, rgba(37,99,235,.16), rgba(255,255,255,0) 60%),
        linear-gradient(135deg, rgba(19,43,85,.10), rgba(255,255,255,1));
        box-shadow: var(--shadow);
        margin-bottom: 14px;
}

.fs-hero h1{
    margin: 0;
    font-size: 34px;
    line-height: 1.15;
    color: var(--navy-900);
}

.fs-hero p{
    margin:6px 0 0 0; 
    color: var(--muted);
}


/*Section chip*/
.fs-chip{
    display:inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid rgba(37,99,235,.18);
    background: rgba(37,99,235,.08);
    color: var(--navy-800);
    font-size: 12px;
    letter-spacing: .2px;
}

/* Generic cards */ 
.fs-card{
    padding: 14px 14px 10px 14px;
    border-radius: 16px;
    border: 1px solid var(--border);
    background: var(--card);
    box-shadow: var(--shadow);
}

/*KPI cards */
.fs-kpi{
    padding: 14px 14px;
    border-radius: 18px;
    border: 1px solid var(--border);
    background:
        radial-gradient(700px 180px  at 10% 0%,rgba(59,130,246,.18), rgba(255,255,255,0) 60%),
        linear-gradient(180deg, #FFFFFF 0%, #F7FAFF 100%);
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
    min-height: 92px;
}

.fs-kpi:before{
    content:"";
    position:absolute;
    inset:0;
    border-radius: 18px;
    background:
        linear-gradient(90deg, rgba(37,99,235,.18), rgba(19,43,85,.05));
    opacity:.35;
    pointer-events:none;
}

.fs-kpi .kpi-top{
    position: relative;
    display:flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}

.fs-kpi .kpi-label{
    font-size: 12px;
    color: var(--muted);
    letter-spacing: .25px;
    text-transform: uppercase;
}

.fs-kpi .kpi-icon{
    width: 34px;
    height: 34px;
    border-radius: 12px;
    background: rgba(37,99,235,.10);
    border: 1px solid rgba(37,99,235,.18);
    display: flex;
    align-items: center;
    font-size: 16px;
    color: var(--navy-800);
}

.fs-kpi .kpi-value{
    position: relative;
    font-size: 26px;
    font-weight: 800;
    color: var(--navy-900);
    line-height: 1.1;
}

.fs-kpi .kpi-sub{
    position: relative;
    margin-top: 6px;
    font-size: 12px;
    color: var(--muted);
}

/* Tabs */ 
button[data-baseweb="tab"]{
    border-radius: 999px !important;
    padding: 8px 14px !important;
    margin-right: 6px !important;
    color: var(--navy-800) !important;
}

button[data-baseweb="tab"][aria-selected="true"]{
    background: rgba(37,99,235,.12) !important;
    border: 1px solid rgba(37,99,235,.25) !important;
}

hr{opacity:.22}
</style>
""", unsafe_allow_html=True)

#KPI card component
def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(f"""
    <div class="fs-kpi">
        <div class="kpi-top">
            <div class="kpi-label">{label}</div>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

#data loading that returns data
@st.cache_data
def load_superstore():
    return pd.read_csv(DATA_DIR / "superstore_clean.csv")

@st.cache_data
def load_forecast_subcategory_inventory():
    fc = pd.read_csv(DATA_DIR / "prophet_forecast_30_days_subcategory.csv")
    
    #cleaning columns
    fc.columns = fc.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    
    #convert date column to datetime
    fc["ds"] = pd.to_datetime(fc["ds"], errors="coerce")
    
    #convert forecast columns to numerics
    for c in ["y", "yhat", "yhat_lower", "yhat_upper"]:
        if c in fc.columns:
            fc[c] = pd.to_numeric(fc[c], errors="coerce")
    
    #making sure required columns exists
    required = {"ds", "yhat", "category", "sub_category"}
    missing = required - set(fc.columns)
    if missing:
        raise ValueError(f"Missing columns in forecast file: {missing}")
    
    #remove invalid rows and sorting by date
    fc = fc.dropna(subset=["ds", "yhat", "category", "sub_category"]).sort_values("ds")
    return fc


#loading data to dataframe
df = load_superstore()

#detecting date column name
if "Order Date" in df.columns:
    date_col = "Order Date"
elif "order_date" in df.columns:
    date_col = "order_date"
else:
    date_col = None

#convert dates
#errors="coerce" turns invalid strings to NaT
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

if date_col and "order_year" not in df.columns:
    df["order_year"] = df[date_col].dt.year
    
#hero reader
st.markdown("""
<div class="fs-hero">
    <h1>ForeSight AI Dashboard</h1>
    <p>Market Trend Prediction & Inventory Intelligence</p>
</div>
""", unsafe_allow_html=True)


#Sidebar fillers
with st.sidebar:
    st.header("Filters")
    st.caption("Use filters to update KPIs and charts in real-time.")
    
    #Date filter
    if date_col:
        #making sure column is datetime
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        
        #computing min/max from data
        valid_dates = df[date_col].dropna()
        if valid_dates.empty:
            st.warning("No valid dates found.")
            start_date = end_date = None
        else:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            
            date_range = st.date_input(
                "Date Range",             
                value=(min_date, max_date),   #default selection
                min_value = min_date,         #earliest date
                max_value = max_date          #latest date
            )
            #converting back to timestapms to filter the dataframe
            start_date = pd.to_datetime(date_range[0]) 
            end_date = pd.to_datetime(date_range[1])
    else:
        st.warning("No data column found.")
        start_date = end_date = None
    
    #category filter
    categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
    selected_category = st.selectbox("Category", categories)
    
    #state filter
    state = ["All"] + sorted(df["state"].dropna().unique().tolist())
    selected_state = st.selectbox("State", state)
    
    #forecast horizon
    forecast_horizon = st.selectbox("Forecast Horizon (Days)", [7, 14, 30 ,60], index = 2)
    
#applying filters
filtered_df = df.copy()

if date_col and "order_year" not in filtered_df.columns:
    filtered_df["order_year"] = filtered_df[date_col].dt.year

#handle the possible column name differences
cat_col = "category" if "category" in filtered_df.columns else "Category"
state_col = "state" if "state" in filtered_df.columns else "State"

#date filter for rows selected between the dates
if date_col and start_date is not None and end_date is not None:
    filtered_df = filtered_df[filtered_df[date_col].between(start_date, end_date)]

#category filter when user selects a category
if selected_category != "All":
    filtered_df = filtered_df[filtered_df[cat_col] == selected_category]

#state filter when user selects a state
if selected_state != "All":
    filtered_df = filtered_df[filtered_df[state_col] == selected_state]
    
    
    
#KPI calculation
total_sales = filtered_df["sales"].sum()
total_profit = filtered_df["profit"].sum()
orders_count = filtered_df["order_id"].nunique()


st.markdown('<span class="fs-chip">Key metrics (filtered)</span>', unsafe_allow_html=True)

#KPI display 
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    kpi_card("Total Sales", f"${total_sales:,.0f}", "Gross revenue")
    
with kpi2:
    kpi_card("Total Profit", f"${total_profit:,.0f}", "Net profit")
    
with kpi3:
    kpi_card("Total Orders", f"{orders_count:,}", "Count of orders")

with kpi4:
    kpi_card("Forecast Horizon", f"{forecast_horizon} Days", "Prediction window")


st.markdown("---")

#tabs

tab1, tab2, tab3, tab4 =st.tabs([
    "Overview",
    "Sales Trends",
    "Forecasting",
    "Inventory Insights"
])


with tab1:
    st.subheader("Overview")
    
    col1, col2 = st.columns(2)
    
    ### sales by category bar chart
    with col1:
        st.markdown("### Total Sales by Category")
        
        #detecting the correct column names
        cat_col = "category" if "category" in filtered_df.columns else "Category"
        sales_col = "sales" if "sales" in filtered_df.columns else "Sales"
        
        #aggregate total sales by category
        category_sales = (
            filtered_df
            .groupby(cat_col)[sales_col]
            .sum()
            .sort_values(ascending=False)
        )
        
        total_sales = category_sales.sum()
        percentages = (category_sales / total_sales) * 100
        
        colors = {
            "Furniture": "dodgerblue",
            "Office Supplies" : "darkcyan",
            "Technology": "orchid"
        }
        
        #list of colors aligned with the category_sales order
        bar_colors = [colors.get(cat, "gray") for cat in category_sales.index]
        
        #building chart
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x = category_sales.index,
                y = category_sales.values,
                text = [f"{pct:.1f}%" for pct in percentages],   #show % above each bar
                textposition='outside',
                marker=dict(color=bar_colors),
                width = 0.6
            )
        )

        fig.update_layout(
            xaxis_title = 'Category',
            yaxis_title = 'Sales ($)',
            height = 400,
            margin = dict(l=15, r=15, t=40, b=15)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        #chrolopeth map
        st.markdown("### Total Sales by State")
        state_abbrev = {
            "Alabama" : "AL","Alaska": "AK","Arizona": "AZ","Arkansas": "AR","California": "CA","Colorado": "CO","Connecticut": "CT","Delaware": "DE","District Of Columbia": "DC","Florida": "FL","Georgia": "GA","Idaho": "ID","Illinois": "IL","Indiana": "IN","Iowa": "IA","Kansas": "KS","Kentucky": "KY","Louisiana": "LA","Maine": "ME","Maryland": "MD","Massachusetts": "MA","Michigan": "MI","Minnesota": "MN","Mississippi": "MS","Missouri": "MO","Montana": "MT","Nebraska": "NE","Nevada": "NV","New Hampshire": "NH","New Jersey": "NJ","New Mexico": "NM","New York": "NY","North Carolina": "NC","North Dakota": "ND","Ohio": "OH","Oklahoma": "OK","Oregon": "OR","Pennsylvania": "PA","Rhode Island": "RI","South Carolina": "SC","South Dakota": "SD","Tennessee": "TN","Texas": "TX","Utah": "UT","Vermont": "VT","Virginia": "VA","Washington": "WA","West Virginia": "WV","Wisconsin": "WI","Wyoming": "WY",
        }
        tmp = filtered_df.copy() #working on a copy to not modify the filtered_df
        state_col = "state" if "state" in tmp.columns else "State"
        
        #standarizing state text format
        tmp[state_col] = tmp[state_col].astype(str).str.strip().str.title()
        
        #create new column with state abbreviations
        tmp["state_code"] = tmp[state_col].map(state_abbrev)
        
        #aggregating sales by state
        state_sales = (
            tmp.dropna(subset=["state_code"])
            .groupby(["state_code", state_col])["sales"]
            .sum()
            .reset_index()
            .rename(columns={state_col: "state"})    #names for hovering
        )
        
        if state_sales.empty:
            st.warning("No state data available")
        else:
            fig = px.choropleth(
            state_sales,
            locations = "state_code",
            locationmode="USA-states",
            color = "sales",
            color_continuous_scale = "Blues",
            scope = "usa",
            labels = {"sales" : "Total Sales ($)"},
            hover_data = {"sales": ":,.0f"},
            hover_name = "state"
        )

        fig.update_layout(
            autosize = True,
            margin=dict(l=0, r=0, t=0, b=0),
            geo = dict(
                bgcolor = "rgba(0,0,0,0)",
                lakecolor = "white",
                fitbounds="locations"
            ),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    

    #total sales and profit (donut chart)
    st.markdown("### Total Sales and Profit by Customer Segment")
    data = filtered_df.copy()
        
    segment_col = "segment" if "segment" in data.columns else "Segment"
    sales_col = "sales" if "sales" in data.columns else "Sales"
    profit_col = "profit" if "profit" in data.columns else "Profit"
        
    #aggregate sales and profit    
    segment_agg = (
            data.groupby(segment_col)[[sales_col, profit_col]]
            .sum()
            .reset_index()
            .sort_values(sales_col, ascending = False)
    )

    if segment_agg.empty:
        st.warning("No segment data available.")
    else:
        segment_color = {
        "Consumer": "darkcyan",
        "Corporate": "dodgerblue",
        "Home Office": "orchid"
    }
    colors_list = [segment_color.get(s, "gray") for s in segment_agg[segment_col]]
            
    fig = make_subplots(
        rows = 1, cols = 2,
        specs = [[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles = ("Sales by Customer Segment", "Profit by Customer Segment")
    )

    #sales donut
    fig.add_trace(
    go.Pie(
        labels = segment_agg[segment_col],
        values = segment_agg[sales_col],
        hole = 0.5,
        marker = dict(colors = colors_list),
        textinfo = "percent+label",
        hovertemplate = ("Segment: %{label} <br>""Sales: $%{value:,.0f}<extra></extra>"),
        name = "Sales"
    ),
    row = 1, col = 1
    )
    
    #profit donut
    fig.add_trace(
        go.Pie(
            labels = segment_agg[segment_col],
            values = segment_agg[profit_col],
            hole = 0.5,
            marker = dict(colors = colors_list),
            textinfo="percent+label",
            hovertemplate = ("Segment: %{label}<br>""Profit: $%{value:,.0f}<extra></extra>"),
            name="Profit"
                ),
            row =1, col = 2
        )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=80, b=70),
        legend=dict(
            title="Customer Segment",
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="left",
            x=0
        ),
    )
    
    #prevent overlapping
    fig.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
            
    fig.update_annotations(font_size=12, y=1.02)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
            
    #total sales by sub-category and category (horizontal bar chart)
    st.markdown("### Total Sales by Sub-Category and Category")
        
    data2 = filtered_df.copy()
        
    cat_col = "category" if "category" in data2.columns else "Category"
    subcat_col = "sub_category" if "sub_category" in data2.columns else "Sub-Category"
    sales2_col = "sales" if "sales" in data2.columns else "Sales"
        
    subcat_sales = (
        data2.groupby([cat_col, subcat_col])[sales2_col]
        .sum()
        .reset_index()
    )

    if subcat_sales.empty:
        st.warning("No category or sub-category available.")
    else:
        colors = {
            "Furniture": "dodgerblue",
            "Office Supplies": "darkcyan",
            "Technology":"orchid"
        }
         
        #dropdown to filter subcategories    
        category_options = ["All Categories"] + sorted(subcat_sales[cat_col].dropna().unique().tolist())
        selected_cat = st.selectbox("Category", category_options, index=0)
            
        
        #aggregating subcategories across all categories
        if selected_cat == "All Categories":  
            plot_df = (
                subcat_sales.groupby(subcat_col, as_index=False)[sales2_col]
                .sum()
                .sort_values(sales2_col, ascending=True)
            )
            bar_colors = ["dodgerblue"] * len(plot_df)
            title = "Total Sales by Sub_Category (All Categories)"
            customdata = ["All"] * len(plot_df)
        else:
            #show subcategories for specific categories
            plot_df = (
                subcat_sales[subcat_sales[cat_col] == selected_cat]
                .sort_values(sales2_col, ascending=True)
            )
            bar_colors = [colors.get(selected_cat, "gray")] *len(plot_df)
            title = f"Total Sales by Sub_Category ({selected_cat})"
            customdata = [selected_cat] * len(plot_df)
    
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x = plot_df[sales2_col],
                y = plot_df[subcat_col],
                orientation = 'h',
                marker = dict (color=bar_colors),
                text = [f"${v:,.0f}" for v in plot_df[sales2_col]],
                textposition = 'outside',
                customdata = customdata,
                hovertemplate = (
                    "Sub-Category: %{y} <br>"
                    "Category: %{customdata} <br>"
                    "Sales: %{x:,.0f} <extra></extra>"
                )
            )
        )
        
        #changing height to fit in the dashboard
        fig2.update_layout(
            title=title,
            xaxis_title="Sales ($)",
            yaxis_title="Sub-Category",
            height=800,
            margin=dict(l=120, r=20, t=70, b=20)
        )
        fig2.update_yaxes(automargin=True)
        st.plotly_chart(fig2, use_container_width=True)

        

with tab2:
    st.subheader("Sales Trends")
    
    #ensuring date_col is datetime                          
    if date_col:
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors="coerce")
    
    #monthly sales per category (line chart)
    st.markdown("### Monthly Sales per Category")
       
    #grouping by month, category, and sales
    monthly_cat =(
        filtered_df.groupby([pd.Grouper(key=date_col, freq="M"), "category"])["sales"]
        .sum()
        .reset_index()
        .rename(columns={date_col: "Date"})
    )
      
    #plot line chart (one line per category)
    fig_cat = px.line(
        monthly_cat,
        x="Date",
        y="sales",
        color="category",
        markers=True
    )
        
    fig_cat.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
        height=420
    )
        
    st.plotly_chart(fig_cat, use_container_width=True)
    st.divider()
    
    #monthly sales trend (overall)
    st.markdown("### Monthly Sales Trend")
        
    #grouping by month and total sales
    monthly_sales = (
        filtered_df.groupby(pd.Grouper(key=date_col, freq="M"))["sales"]
        .sum()
        .reset_index()
        .rename(columns={date_col: "Date"})
    )
    
    #convert date to datetime
    monthly_sales["Date"] = pd.to_datetime(monthly_sales["Date"], errors="coerce")
        
    #plotting overall sales trend line
    fig_trend = px.line(
        monthly_sales,
        x="Date",
        y="sales",
        markers=True,
        line_shape="spline"
    )
        
    fig_trend.update_layout(
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified",
        height=420
    )
        
    st.plotly_chart(fig_trend, use_container_width = True)
    st.divider()
    
    #total profit by category and cub-category
    st.markdown("### Total Profit by Category & Sub-Category")
    
    #filtering row
    f1, f2, f3 = st.columns([1.2,2.0, 1.0], gap="large")
    
    #choose whether to view category or subcategory
    view_mode = f1.radio("View", ["By Category", "By Sub-Category"], horizontal=True)
    
    #filtering categories
    categories = sorted(df["category"].dropna().unique().tolist())
    selected_category = f2.selectbox("Category filter", ["All"] + categories, index=0)
    
    #top n sliders how many bars there are
    top_n = f3.slider("Top N", min_value=5, max_value=25, value=10, step=1)
    
    #working on a copy for profit aggregation
    df_profit = df.copy()
    if selected_category !="All":
        df_profit = df_profit[df_profit["category"] == selected_category]
        
    #aggregating profit by category
    cat_profit = (
        df_profit.groupby("category")["profit"]
        .sum()
        .reset_index()
        .sort_values("profit", ascending=False)
        .head(top_n)
    )
    
    #aggregate profit by category and sub_category
    subcat_profit = (
        df_profit.groupby(["category", "sub_category"])["profit"]
        .sum()
        .reset_index()
        .sort_values("profit", ascending=False)
        .head(top_n)
    )
    
    #plot depending on selected view mode
    if view_mode == "By Category":
        #horizontal bar chart profit by category
        fig_profit = px.bar(
            cat_profit.sort_values("profit", ascending=True),
            x="profit",
            y="category",
            orientation="h",
            text="profit"
        )
        fig_profit.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        
        
        fig_profit.update_layout(
            xaxis_title="Profit ($)",
            yaxis_title="Category",
            height = 520,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
    else:
        #horizontal bar chart profit by sub_category, colored by category
        fig_profit = px.bar(
            subcat_profit.sort_values("profit", ascending=True),
            x="profit",
            y="sub_category",
            color="category",
            orientation="h",
            text="profit"
        )
        fig_profit.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        
        fig_profit.update_layout(
            xaxis_title="Profit ($)",
            yaxis_title="Category",
            height = 520,
            legend_title_text="",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
    st.plotly_chart(fig_profit, use_container_width=True)
        
    #show underlying aggregated data in a table for transparency    
    with st.expander("Show profit table"):
        if view_mode== "By Category":
            st.dataframe(cat_profit, use_container_width=True)
        else:
            st.dataframe(subcat_profit, use_container_width=True)

with tab3:
    st.subheader("Forecasting")
    
    #load data
    @st.cache_data
    def load_forecast_subcategory(path: Path) -> pd.DataFrame:
        #load prophet forecast output
        fc = pd.read_csv(path)
        
        #cleaning columns
        fc.columns = fc.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

        #convert prophet "ds" (date) column to datetime
        fc["ds"] = pd.to_datetime(fc["ds"], errors = "coerce")
        
        #convert numeric columns to numbers 
        for c in["y", "yhat", "yhat_lower", "yhat_upper"]:
            if c in fc.columns:
                fc[c] = pd.to_numeric(fc[c], errors="coerce")
        
        #validate required columns exist
        required = {"ds", "y", "yhat", "yhat_lower", "yhat_upper", "category", "sub_category"}
        missing = required - set(fc.columns)
        if missing: 
            raise ValueError(f"Missing columns in forecast file: {missing}")
        
        #drop rows with missing values and sort by date
        fc = fc.dropna(subset=["ds", "yhat", "category", "sub_category"]).sort_values("ds")
        return fc
    
    #test merge
    @st.cache_data
    def load_test_merge():
        tm = pd.read_csv(DATA_DIR / "prophet_test_merge.csv")
        tm["ds"] = pd.to_datetime(tm["ds"], errors="coerce")
        
        for c in ["y", "yhat"]:
            if c in tm.columns:
                tm[c] = pd.to_numeric(tm[c], errors="coerce")
        
        tm = tm.dropna(subset=["ds", "y", "yhat"]).sort_values("ds")
        return tm
    
    
    #paths
    forecast_path = DATA_DIR / "prophet_forecast_30_days_subcategory.csv"
    
    #load forecasting
    try: 
        fc = load_forecast_subcategory(forecast_path)
    except Exception as e: 
        st.error("Could not load forecast CSV file.")
        st.exception(e)
        st.stop()
        
    #load test merge
    try:
        test_merge = load_test_merge()
    except Exception as e:
        st.warning("Could not load test_merge CSV file.")
        test_merge = pd.DataFrame()
    
    
    #filters
    st.markdown("##### Filters")
    
    f1, f2, f3 = st.columns([2, 2, 1])
    
    #category options
    cat_options = ["All Sales"] + sorted(fc["category"].dropna().unique().tolist())
    
    with f1:
        chosen_cat = st.selectbox("Category", cat_options, index=0, key="fc_cat")
        
    #subcategory options
    with f2:
        if chosen_cat == "All Sales":
            #disabling dropdown depending on chosen category
            chosen_subcat = "All Sub-Categories"
            st.selectbox("Sub-Category", ["All Sub-Categories"], index=0, key="fc_subcat_disabled", disabled=True)
        else:
            subcats = sorted(fc.loc[fc["category"] == chosen_cat, "sub_category"].dropna().unique().tolist())
            chosen_subcat = st.selectbox(
                "Sub-Category",
                ["All Sub-Categories"] + subcats,
                index=0,
                key="fc_subcat"
            )
            
    with f3:
        #number of forecast days to display
        horizon = st.slider("Show next N days", 7, 30, 30, key="fc_horizon")
        
    if chosen_cat == "All Sales":
        #aggregate forecast totals across all categories/subcategories per day
        series_df = (
            fc.groupby("ds", as_index=False)[["yhat", "yhat_lower", "yhat_upper"]]
            .sum()
            .sort_values("ds")
        )
        series_df["category"] = "All Sales"
        series_df["sub_category"] = "All Sub-Categories"
        series_df["y"] = np.nan
    
    else:
        #filter forecast data
        cat_df = fc[fc["category"] == chosen_cat].copy()
        
        if chosen_subcat == "All Sub-Categories":
            #sum of subcategories to get category total forecast per day
            series_df = (
                cat_df.groupby("ds", as_index=False)[["yhat", "yhat_lower", "yhat_upper"]]
                .sum()
                .sort_values("ds")
            )
            series_df["category"] = chosen_cat
            series_df["sub_category"] = "All Sub-Categories"
            series_df["y"] = np.nan
        else:
            #choosing specific subcategory time series
            series_df = (
                cat_df[cat_df["sub_category"] == chosen_subcat]
                .sort_values("ds")
                .copy()
            )
            
            #safety net to ensure that y exist in the table
            if "y" not in series_df.columns:
                series_df["y"] = np.nan
        
    #stop if there data is empty
    if series_df.empty:
        st.warning("No forecast rows found for this selection.")
        st.stop()
        
    #showing the last N forecast days
    series_df = series_df.tail(horizon).copy()
        
    
    #KPI
    c1, c2, c3, c4, c5 = st.columns(5)
    
    total_forecast = series_df["yhat"].sum()
    avg_daily = series_df["yhat"].mean()
    peak_idx = series_df["yhat"].idxmax()
    peak_day = series_df.loc[peak_idx, "ds"].date()
    avg_unc = (series_df["yhat_upper"] - series_df["yhat_lower"]).mean()
    
    c1.metric("Forecast Total", f"${total_forecast:,.0f}")
    c2.metric("Average Daily Forecast", f"${avg_daily:,.0f}")
    c3.metric("Peak Day", f"{peak_day}")
    c4.metric("Average Uncertainty Band", f"${avg_unc:,.0f}")
    
    st.divider()
    
    
    #forecast chart with uncertainty band
    title = chosen_cat
    if chosen_cat != "All Sales" and chosen_subcat != "All Sub-Categories":
        title += f" - {chosen_subcat}"
    elif chosen_cat != "All Sales":
        title += " - All Sub-Categories"
    
    st.markdown(f"### Forecast: **{title}** (Next 30 Days)")
    
    fig = go.Figure()
    
    #uncertainty band
    fig.add_trace(go.Scatter(
        x=pd.concat([series_df["ds"], series_df["ds"][::-1]]),
        y=pd.concat([series_df["yhat_upper"], series_df["yhat_lower"][::-1]]),
        fill="toself",
        line=dict(width=0),
        name="Uncertainty",
        hoverinfo="skip"
    ))
    
    #yhat line
    fig.add_trace(go.Scatter(
        x=series_df["ds"],
        y=series_df["yhat"],
        mode="lines+markers",
        name="Forecast (yhat)",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y:,.0f}"
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    #actual vs predicted graph
    st.markdown("### Prophet Evaluation: Actual vs Predicted (Last 30 Days)")
    
    if not test_merge.empty:
        fig_eval = go.Figure()
        
        fig_eval.add_trace(go.Scatter(
            x = test_merge["ds"],
            y = test_merge["y"],
            mode = "lines+markers",
            name="Actual"
        ))
        
        fig_eval.add_trace(go.Scatter(
            x = test_merge["ds"],
            y = test_merge["yhat"],
            mode="lines+markers",
            name="Predicted"
        ))
        
        fig_eval.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales",
            hovermode="x unified",
            height=420
        )
        
        st.plotly_chart(fig_eval, use_container_width=True)
    else:
        st.warning("No evaluation data available")
        
    st.divider()
            
    
    
    
    
    #ai insight
    st.markdown("### AI Forecast Insight")
    
    avg_last_30 = series_df["yhat"].tail(min(30, len(series_df))).mean()
    forecast_next_n = series_df["yhat"].sum()
    trend = "upward" if forecast_next_n > avg_last_30 else "downward"
    
    summary = {
        "category": chosen_cat,
        "sub_category": chosen_subcat,
        "avg_sales_last_30": round(avg_last_30, 2),
        "forecast_total_next_period": round(forecast_next_n, 2),
        "trend": trend,
        "horizon_days": horizon,
        "average_uncertainty_band": round(avg_unc, 2)
    }
    
    if st.button("Generate AI Insights", key="generate_ai_forecast_insight"):
        with st.spinner("Generating AI insight..."):
            insight = generate_ai_insight(summary)
        st.info(insight)
    
    st.divider()
        
                            
    
    #category comparison and next N days total
    
    if chosen_cat == "All Sales":
        st.markdown("### Category Comparison (Next Days Total)")

        
        #for caegory and sub-categor, take the last "horizon" days
        subcat_window = (
            fc.groupby(["category", "sub_category"], group_keys=False)
            .apply(lambda g: g.sort_values("ds").tail(horizon))
        )
        
        #sum csubcategory totals into category totals
        bycat_totals = (
            subcat_window.groupby("category", as_index=False)["yhat"]
            .sum()
            .sort_values("yhat", ascending=False)
        )
        
        st.bar_chart(bycat_totals.set_index("category")["yhat"]) 
        st.divider()
        
    #forecast table and download
    st.markdown("### Forecast Table")
    
    table_cols = ["ds","y", "yhat", "yhat_lower", "yhat_upper", "category", "sub_category"]
    for c in table_cols:
        if c not in series_df.columns:
            series_df[c] = np.nan
    
    st.dataframe(series_df[table_cols], use_container_width=True)
    
    st.download_button(
        "Download this forecast (CSV)",
        data=series_df.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast{chosen_cat.replace(' ', '_').lower()}.csv",
        mime="text/csv"
    )
    
with tab4:
    st.subheader("Inventory Insights")
    st.caption("Quick summary for high-demand products and inventory decision support.")
    
    #loading forecast data
    try:
        fc_inventory = load_forecast_subcategory_inventory()
    except Exception as e:
        st.error("Could not load forecast data for inventory insights. ")
        st.exception(e)
        st.stop()
    
    
    #streamlit rerun script on every interaction
    #using session_state to persist which buttons were clicked
    st.session_state.setdefault("show_top_products", False)
    st.session_state.setdefault("run_low_stock", False)
    st.session_state.setdefault("gen_suggestions", False)
    
    #using filtered data for sidebars to take affect
    df_use = filtered_df.copy()
    
    #detect date column for the recent sales calculation
    inv_date_col = None
    if "order_date" in df_use.columns:
        inv_date_col = "order_date"
    elif "Order Date" in df_use.columns:
        inv_date_col = "Order Date"
    
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.markdown("### High Demand Items")
        st.write("View top-selling products based on historical sales.")
        if st.button(
            "View Top Products", 
            key="btn-top_products",
            use_container_width=True
        ):
            st.session_state["show_top_products"] = True
        
    with colB:
        st.markdown("### Low Stock Alerts")
        st.write("Detect items with the strongest forecasted demand.")
        if st.button(
            "Run Low Stock Check",
            key="btn_low_stock",
            use_container_width=True
        ):
            st.session_state["run_low_stock"] = True
        
    with colC:
        st.markdown("### Stock Recommendations")
        st.write("Generate restocking suggestions based on demand trends.")
        if st.button(
            "Generate Suggestions", 
            key="btn_gen_suggestions",
            use_container_width=True
        ):
            st.session_state["gen_suggestions"] = True
            
    #reset button to clear all output
    if st.button("Reset", type="secondary"):
        st.session_state["show_top_products"] = False
        st.session_state["run_low_stock"] = False
        st.session_state["gen_suggestions"] = False
    
    st.markdown("---")
    
    #Top products
    
    #view top products
    if st.session_state["show_top_products"]:
        
        st.markdown("### Top 10 Best-Selling Products (Lollipop Chart)")
        st.caption("Products ranked by total sales revenue.")

        #aggregating total sales per product and take top 10 products
        top10_products = (
            df_use.groupby("product_name")["sales"]
            .sum()
            .sort_values(ascending = False)
            .head(10)
            .reset_index()
        )
        #reserving order so highest value is on top
        top10_products = top10_products.iloc[::-1]

        #creating lollipop chart
        fig = go.Figure()

        #stick lines (lollipop chart)
        fig.add_trace(
            go.Scatter(
                x = top10_products["sales"],
                y = top10_products["product_name"],
                mode = "lines",
                line = dict(color="lightgray", width = 3),
                showlegend = False,
                hoverinfo = "skip"
            )
        )

        #adding lollipop head markers and labels
        fig.add_trace(
            go.Scatter(
                x = top10_products["sales"],
                y = top10_products["product_name"],
                mode = "markers+text",
                marker = dict(size=14, color="dodgerblue"),
                text = [f"${v:,.0f}" for v in top10_products["sales"]],
                textposition = "middle right",
                textfont = dict(size = 12),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Sales: $%{x:,.0f}<extra></extra>"
                ),
                showlegend = False
            )
        )

        #layout for dashboard
        fig.update_layout(
            xaxis_title = "Sales ($)",
            yaxis_title = "Product",
            height = 700,
            margin = dict(
                l = 250,
                r = 40,
                t = 60,
                b= 40)
        )

        fig.update_yaxes(automargin=True)

        st.plotly_chart(fig, use_container_width=True, key="top10_lollipop_chart")

        with st.expander("Show Top 10 Products Table"):
            st.dataframe(
                top10_products.sort_values("sales", ascending=False).reset_index(drop=True), use_container_width=True
            )
        
        st.divider()

        
        
    #low stock check (demand based) since there is no inventory
    if st.session_state["run_low_stock"]:
        st.subheader("Low Stock Alerts (Forecast-Based)")
        st.caption("Potential items in need of replenishment based on demand.")
        
        #category filter for forecast alert
        forecast_cat_options = ["All"] + sorted(fc_inventory["category"].dropna().unique().tolist())
        chosen_alert_cat = st.selectbox(
            "Category Filter",
            forecast_cat_options,
            key="low_stock_category_filter"
        )
        
        #forecast window
        alert_days = st.slider(
            "Forecast Window (Days)",
            min_value = 7,
            max_value = 30,
            value = 14,
            step = 1,
            key = "low_stock_days"
        )
        
        alert_fc = fc_inventory.copy()
        
        #apply category filter if selected
        if chosen_alert_cat != "All":
            alert_fc = alert_fc[alert_fc["category"] == chosen_alert_cat]
        
        #keeping the first N future dates from forecast
        available_dates = sorted(alert_fc["ds"].dropna().unique())
        selected_dates = available_dates[:alert_days]
        
        alert_window = alert_fc[alert_fc["ds"].isin(selected_dates)]
        
        if alert_window.empty:
            st.warning("No forecast rows found for the selected filter." )
        else:
            #aggregating forecasted sales by category and sub-category
            alert_summary = (
                alert_window.groupby(["category", "sub_category"], as_index=False)["yhat"]
                .sum()
                .rename(columns={"yhat" : "forecast_sales"})
                .sort_values("forecast_sales", ascending=False)
            )
            
            high_threshold = alert_summary["forecast_sales"].quantile(0.85)
            med_threshold = alert_summary["forecast_sales"].quantile(0.60)
            
            #classifying alert level
            def classify_alert(x):
                if x >= high_threshold:
                    return "High Alert"
                elif x >= med_threshold:
                    return "Medium Alert"
                return "Low Alert"
            
            alert_summary["alert_level"] = alert_summary["forecast_sales"].apply(classify_alert)
            
            #show result table
            st.dataframe(
                alert_summary.reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
            
            st.info(
                "High-alert rows indicate category and sub-category that has the highest demand."
            )
            
        st.divider()
            
        
       
    #generate recommendations based on sales
    
    if st.session_state["gen_suggestions"]:
        st.subheader("Recommendations (Sales-Based)")
        st.caption("Recommendations based on historical sales, recent demand, and forecasted demand. ")
        
        #historical sales by category and sub-category
        hist_sales = (
            df_use.groupby(["category", "sub_category"], as_index=False)["sales"]
            .sum()
            .rename(columns={"sales": "historical_sales"})
        )
        
        #recent sales by category and sub-category
        if inv_date_col is not None:
            days_recent = st.slider(
                "Recent Sales Window (Days)",
                min_value = 7,
                max_value = 120,
                value = 30,
                step = 7,
                key = "rec_recent_days"
            )
            
            df_recent = df_use.dropna(subset=[inv_date_col]).copy()
            end_recent = df_recent[inv_date_col].max()
            start_recent = end_recent - pd.Timedelta(days=days_recent)
            
            recent = df_recent[
                (df_recent[inv_date_col] >= start_recent) &
                (df_recent[inv_date_col] <= end_recent)
            ]
            
            recent_sales = (
                recent.groupby(["category", "sub_category"], as_index=False)["sales"]
                .sum()
                .rename(columns={"sales": "recent_sales"})
            )
        else:
            recent_sales = hist_sales[["category", "sub_category"]].copy()
            recent_sales["recent_sales"] = 0
        
        #forecast sales by category and sub-category
        rec_days = st.slider(
            "Forecast Window (Days)",
            min_value = 7,
            max_value = 30,
            value = 14,
            step = 1,
            key = "rec_forecast_days"
        )
        
        rec_dates = sorted(fc_inventory["ds"].dropna().unique())[:rec_days]
        
        forecast_sales = (
            fc_inventory[fc_inventory["ds"].isin(rec_dates)]
            .groupby(["category", "sub_category"], as_index=False)["yhat"]
            .sum()
            .rename(columns={"yhat": "forecast_sales"})
        )
        
        #merging historical, recent, and forecast data
        recs = hist_sales.merge(recent_sales, on=["category", "sub_category"], how="left")
        recs = recs.merge(forecast_sales, on=["category", "sub_category"], how="left")
            
        recs["recent_sales"] = recs ["recent_sales"].fillna(0)
        recs["forecast_sales"] = recs["forecast_sales"].fillna(0)
            
        #weighted score:
        #40% on historical demand
        #30% on recent demand
        #30% on forecast demand
        recs["priority_score"] = (
            0.4 * recs["historical_sales"] +
            0.3 * recs["recent_sales"] +
            0.3 * recs["forecast_sales"]
        )
        
        #defining recommendation bands
        high_cut = recs["priority_score"].quantile(0.85)
        med_cut = recs["priority_score"].quantile(0.60)
        
        def recommendation_label(score):
            if score >= high_cut:
                return "Restock Now"
            elif score >= med_cut:
                return "Monitor Closely"
            return "Stable Demand"
        
        recs["recommendation"] = recs["priority_score"].apply(recommendation_label)
        
        #sort by highest priority
        recs = recs.sort_values("priority_score", ascending=False).reset_index(drop=True)
        
        show_n = st.slider(
            "Show Top N Recommendations",
            min_value = 5,
            max_value = 30,
            value = 12,
            step = 1,
            key = "show_n_recommendations"
        )
        
        st.dataframe(
            recs.head(show_n),
            use_container_width=True,
            hide_index=True
        )
        
        if not recs.empty:
            top_row = recs.iloc[0]
            
            suggestion_summary = {
                "category" : top_row["category"],
                "sub-category" : top_row["sub_category"],
                "historical_sales" : round(top_row["historical_sales"], 2),
                "recent_sales" : round(top_row["recent_sales"], 2),
                "forecast_sales" : round(top_row["forecast_sales"], 2),
                "priority_score" : round(top_row["priority_score"], 2),
                "recommendation" : top_row["recommendation"]
            }
            
            if st.button("Explain Top Recommendation with AI", key="ai_inventory_explain"):
                with st.spinner("Generating AI recommendation insight..."):
                    ai_text = generate_ai_insight(suggestion_summary)
                st.info(ai_text)
                st.success("Recommendations generated using historical sales, recent sales, and forecasted demand.")
                st.divider()
        
        
             
    #top products by year
    
    st.markdown("### Top 10 Best Selling Products by Year (Lollipop)")

    #aggregate sakes per year and product
    year_product_sales = (
        df.groupby(['order_year', 'product_name'])['sales']
        .sum()
        .reset_index()
    )

    #returns top 10 products by sales
    def get_top10_for_year(year):
        temp = (
            year_product_sales[year_product_sales['order_year'] == year]
            .sort_values('sales', ascending = False)
            .head(10)
            .reset_index(drop=True)
        )
        return temp.iloc[::-1]
    
    #list of available years 
    years = sorted(year_product_sales['order_year'].unique())

    #choose the initial year shown in the animation
    initial_year = years[0]
    top10_init = get_top10_for_year(initial_year)

    fig = go.Figure()

    #stick lines
    fig.add_trace(
        go.Scatter(
            x = top10_init['sales'],
            y = top10_init['product_name'],
            mode = 'lines',
            line = dict(color='lightgray', width =3),
            showlegend=False,
            hoverinfo='skip'
        )
    )

    #lollipops trace
    fig.add_trace(
        go.Scatter(
            x=top10_init['sales'],
            y=top10_init['product_name'],
            mode='markers+text',
            marker = dict(size = 14, color = 'dodgerblue'),
            text=[f"${v:,.0f}" for v in top10_init['sales']],
            textposition = 'middle right',
            textfont = dict(size=12),
            hovertemplate = "<b>%{y}</b><br>Sales: $%{x:,.0f}<extra></extra>",
            showlegend = False
        )
    )



    #build animation frames (one frame per year)
    frames = []

    for yr in years:
        top10_year = get_top10_for_year(yr)

        frames.append(
            go.Frame(
                data = [
                    #stick lines
                    go.Scatter(
                        x = top10_year['sales'],
                        y = top10_year['product_name'],
                        mode = 'lines',
                        line = dict(color = 'lightgray', width = 3),
                        showlegend = False,
                        hoverinfo = 'skip'
                    ),
                    #lollipops
                    go.Scatter(
                        x = top10_year['sales'],
                        y = top10_year['product_name'],
                        mode = 'markers+text',
                        marker = dict(size=14, color='dodgerblue'),
                        text=[f"${v:,.0f}" for v in top10_year['sales']],
                        textposition = 'middle right',
                        textfont = dict(size = 12),
                        hovertemplate = "<b>%{y}</b><br>Sales: $%{x:,.0f}<extra></extra>",
                        showlegend = False
                    )
                ],
                name = str(yr) #frame name used by slider and buttons
            )
        )
    fig.frames = frames

    #adding play/pause buttons and slider between years
    fig.update_layout(
        title = f"Top 10 Best-Selling Products by Year (Lollipop)",
        xaxis_title = "Sales ($)",
        yaxis_title = "Product",
        height = 700,
        margin = dict(
            l = 250,
            r = 80,
            t = 80,
            b = 60
        ),
        #buttons (play/pause)
        updatemenus=[
            dict(
                type="buttons",
                showactive = False,
                x = 1.15,
                y = 1.05,
                buttons = [
                    dict(
                        label = "Play",
                        method = "animate",
                        args = [
                            None,
                            {
                                "frame": {"duration": 800, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300}
                            }
                        ]
                    ),
                    dict(
                        label = "Pause",
                        method = "animate",
                        args = [
                            [None],
                            {"frame": {"duration": 0, "redraw": False},
                             "mode": "immediate"}
                        ]
                    )
                ]
            )
        ],
        #slider (select a year)
        sliders = [
            dict(
                active = 0,
                x = 0.1,
                y = -0.05,
                len = 0.8,
                steps = [
                    dict(
                        label = str(yr),
                        method = "animate",
                        args = [
                            [str(yr)],
                            {"frame": {"duration": 500, "redraw": True},
                             "mode": "immediate"}
                        ]
                    )
                    for yr in years
                ]
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True, key="lollipop_year_animation")



