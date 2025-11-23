import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dynamic Pricing Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main container styling */
    .main { background-color: #f8fafc; }
    
    /* Metric Cards */
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 5px solid #3b82f6;
    }
    .metric-label { font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 2rem; color: #1e293b; font-weight: 700; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; }
    .positive { color: #16a34a; }
    
    /* Insight Box */
    .insight-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }
    
    /* Price Tags */
    .price-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .bundle-highlight {
        background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%);
    }
    .price-title { font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px; }
    .price-tag { font-size: 1.8rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def load_data():
    """
    Loads data from the uploaded file or the default Sankalp.xlsx.
    Handles cases where .xlsx files might actually be CSVs.
    """
    uploaded_file = st.sidebar.file_uploader("Upload WTP Data (Optional)", type=['xlsx', 'csv'])
    
    df = None
    
    # 1. Try Uploaded File
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                try:
                    df = pd.read_excel(uploaded_file)
                except:
                    # Fallback: Try reading uploaded xlsx as csv
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
            st.sidebar.success("Custom file loaded!")
        except Exception as e:
            st.sidebar.error(f"Error loading uploaded file: {e}")

    # 2. Try Default File (Sankalp.xlsx)
    if df is None:
        default_file = "Sankalp.xlsx"
        try:
            # Attempt 1: Read as Standard Excel
            try:
                df = pd.read_excel(default_file)
            except Exception:
                # Attempt 2: Read as CSV (even if named .xlsx)
                # This fixes the specific error where a CSV is saved with an xlsx extension
                try:
                    df = pd.read_csv(default_file)
                except:
                     # Attempt 3: Try looking for the .csv filename variant
                    df = pd.read_csv("Sankalp.csv")
            
            st.sidebar.info(f"Using default dataset: {default_file}")
            
        except Exception as e:
            st.error(f"Could not load data. Please upload a file manually. (Error: {e})")
            return None

    return df

def revenue_objective(price, wtp_array):
    """
    Objective function for Differential Evolution.
    Returns NEGATIVE revenue (since DE minimizes).
    """
    p = price[0]
    # Demand is the number of customers willing to pay >= price p
    demand = np.sum(wtp_array >= p)
    revenue = p * demand
    return -revenue 

def get_optimal_price(wtp_series):
    """
    Runs Differential Evolution to find the optimal price.
    """
    # Clean data: drop NaNs and ensure numeric
    wtp_values = pd.to_numeric(wtp_series, errors='coerce').dropna().values
    
    if len(wtp_values) == 0:
        return 0, 0, 0
        
    min_price = float(wtp_values.min())
    max_price = float(wtp_values.max())
    
    # If all prices are the same or 0
    if min_price == max_price:
        return min_price, min_price * len(wtp_values), len(wtp_values)
    
    bounds = [(min_price, max_price)]
    
    result = differential_evolution(
        revenue_objective, 
        bounds, 
        args=(wtp_values,),
        strategy='best1bin', 
        maxiter=100, 
        popsize=15, 
        tol=0.01, 
        mutation=(0.5, 1), 
        recombination=0.7
    )
    
    opt_price = result.x[0]
    opt_demand = np.sum(wtp_values >= opt_price)
    opt_revenue = opt_price * opt_demand
    
    return opt_price, opt_revenue, opt_demand

def generate_demand_curve(df, products, opt_prices):
    """
    Generates a synthetic demand curve for the Bundle.
    """
    # Create a Bundle WTP column
    bundle_wtp = df[products].sum(axis=1)
    
    min_p = bundle_wtp.min() * 0.5
    max_p = bundle_wtp.max()
    
    price_range = np.linspace(min_p, max_p, 100)
    demands = []
    
    for p in price_range:
        demands.append(np.sum(bundle_wtp >= p))
        
    return pd.DataFrame({"Price": price_range, "Demand": demands})

# --- MAIN APP LOGIC ---

st.title("Dynamic Pricing Optimization Dashboard")
st.markdown("Optimization based on Customer **Willingness to Pay (WTP)** data using **Differential Evolution**.")

df = load_data()

if df is not None:
    # Validate columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.error("The dataset contains no numeric columns for pricing.")
    else:
        # Sidebar controls
        st.sidebar.header("Configuration")
        selected_products = st.sidebar.multiselect(
            "Select Products to Analyze", 
            numeric_cols, 
            default=numeric_cols[:5] if len(numeric_cols) >=5 else numeric_cols
        )
        
        if selected_products:
            # --- SECTION 1: OVERVIEW METRICS ---
            st.subheader("1. Optimization Overview")
            
            # Calculate Bundle WTP
            df['Bundle_WTP'] = df[selected_products].sum(axis=1)
            
            # Optimize Individual Products
            results = {}
            total_projected_revenue = 0
            
            progress_bar = st.progress(0)
            for i, prod in enumerate(selected_products):
                p_opt, rev_opt, dem_opt = get_optimal_price(df[prod])
                results[prod] = {
                    "price": p_opt,
                    "revenue": rev_opt,
                    "demand": dem_opt
                }
                total_projected_revenue += rev_opt
                progress_bar.progress((i + 1) / len(selected_products))
            progress_bar.empty()
            
            # Optimize Bundle
            bundle_opt_price, bundle_max_revenue, bundle_demand = get_optimal_price(df['Bundle_WTP'])
            
            # Top Metrics Row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Total Potential Revenue (Individual)</div>
                    <div class="metric-value">₹{total_projected_revenue:,.0f}</div>
                    <div class="metric-delta positive">Based on Optimal Prices</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Optimal Bundle Revenue</div>
                    <div class="metric-value">₹{bundle_max_revenue:,.0f}</div>
                    <div class="metric-delta positive">If sold as All-in-One</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                revenue_lift = ((bundle_max_revenue - total_projected_revenue) / total_projected_revenue) * 100
                color_class = "positive" if revenue_lift >= 0 else "negative"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Bundle Strategy Impact</div>
                    <div class="metric-value">{revenue_lift:+.1f}%</div>
                    <div class="metric-delta {color_class}">Revenue Lift vs Individual</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("---\n")
            
            # --- SECTION 2: OPTIMAL PRICING CARDS ---
            st.subheader("2. Recommended Pricing Strategy")
            st.markdown("Below are the AI-optimized price points derived from the differential evolution algorithm on the WTP dataset.")
            
            # Create a flexible grid for price cards
            cols = st.columns(len(selected_products) + 1)
            
            # Individual Product Cards
            for idx, prod in enumerate(selected_products):
                p_opt = results[prod]['price']
                with cols[idx]:
                    st.markdown(f"""
                    <div class="price-card">
                        <div class="price-title">{prod.replace('_', ' ')}</div>
                        <div class="price-tag">₹{p_opt:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Bundle Price Card
            with cols[-1]:
                st.markdown(f"""
                <div class="price-card bundle-highlight">
                    <div class="price-title">ALL-IN BUNDLE</div>
                    <div class="price-tag">₹{bundle_opt_price:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- SECTION 3: VISUALIZATION ---
            st.write("---\n")
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("3. Product Price Elasticity")
                # Histogram of WTP for first selected product
                target_prod = st.selectbox("Select Product to View WTP Distribution", selected_products)
                fig_hist = px.histogram(
                    df, 
                    x=target_prod, 
                    nbins=30,
                    title=f"Willingness to Pay Distribution: {target_prod}",
                    labels={target_prod: "Price Willing to Pay (₹)"},
                    color_discrete_sequence=['#6366f1']
                )
                fig_hist.add_vline(x=results[target_prod]['price'], line_dash="dash", line_color="green", annotation_text="Optimal Price")
                fig_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.info(f"The dashed green line shows the optimal price (₹{results[target_prod]['price']:,.0f}) relative to customer valuations.")

            with c2:
                st.subheader("4. Bundle Demand Sensitivity")
                # Demand Curve for Bundle
                demand_data = generate_demand_curve(df, selected_products, results)
                
                fig_line = px.line(
                    demand_data, x="Price", y="Demand",
                    title="Projected Bundle Sales at Different Price Points",
                    labels={"Price": "Bundle Price (₹)", "Demand": "Number of Buyers"}
                )
                
                # Add vertical line for optimal price
                fig_line.add_vline(x=bundle_opt_price, line_dash="dash", line_color="green", annotation_text="Optimal Bundle Price")
                fig_line.update_layout(hovermode="x unified")
                fig_line.update_traces(line_color='#ec4899', fill='tozeroy', fillcolor='rgba(236, 72, 153, 0.1)')
                
                st.plotly_chart(fig_line, use_container_width=True)
                st.info("The curve shows how many customers would buy the bundle as the price increases. The peak revenue is at the optimal price point.")

            # --- SECTION 5: DATA PREVIEW ---
            with st.expander("View Raw Data Analysis"):
                st.dataframe(df[selected_products + ['Bundle_WTP']].head(10))
                st.download_button("Download Processed Data", df.to_csv(index=False), "processed_pricing_data.csv")

else:
    st.info("Waiting for data... Please ensure 'Sankalp.xlsx' is in the directory or upload a file.")
