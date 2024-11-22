# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:14:13 2024

@author: 310223340

streamlit run ./LeadAnalysis/lead_app.py
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import scipy.stats as stats
import plotly
import altair as alt
import datetime
from pathlib import Path

#%%set page configuration
st.set_page_config(
    page_title="Lead Data Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Custom CSS to resize the selectbox
st.markdown("""
    <style>
    .stSelectbox {
        width: 33% !important;
    }
    </style>
""", unsafe_allow_html=True)



#%%Define functions to wrap labels and load data
# Function to wrap labels
def wrap_labels(labels, max_length=25):
    wrapped_labels = []
    for label in labels:
        wrapped_label = '<br>'.join([label[i:i+max_length] for i in range(0, len(label), max_length)])
        wrapped_labels.append(wrapped_label)
    return wrapped_labels

# Helper function to format Euro values
def format_euro(value):
    if value >= 1e6:
        return f'€{value/1e6:.1f}M'
    elif value >= 1e3:
        return f'€{value/1e3:.1f}K'
    else:
        return f'€{value:.1f}'

# Load your dataframe
@st.cache_data
def load_data():
    #with open('./LeadAnalysis/lead_app_data.pkl', 'rb') as file:
    with open(Path(__file__).parent/'data/lead_app_data_v2.pkl', 'rb') as file:
        loaded_dataframes = pickle.load(file)
    return loaded_dataframes

# Define a function to create time series plots
def create_time_series(data, time_column, value_column, title, yaxis_title, color_column=None):
    """
    Function to create a time series plot using Plotly.

    Parameters:
    - data: DataFrame containing the data
    - time_column: Name of the column representing time (e.g., Date, Month)
    - value_column: Name of the column representing the values to plot (e.g., Lead Count)
    - title: Title of the plot
    - yaxis_title: Label for the y-axis
    - color_column: Optional, name of the column to color by category (for multiple time series)
    """
    # Create the time series plot
    fig = px.line(data, 
                  x=time_column, 
                  y=value_column, 
                  color=color_column,  # Optional: Use if plotting multiple time series
                  title=title, 
                  labels={time_column: 'Time', value_column: yaxis_title})
    
    # Update layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title=yaxis_title,
        height=600,
        template='plotly_dark'  # Suitable for black background
    )
    
    # Return the figure
    return fig

# Define a function to create a stacked bar chart for time series data
def create_stacked_bar_ts(data, time_column, value_column, title, yaxis_title, color_column):
    """
    Function to create a stacked bar chart for time series data using Plotly.

    Parameters:
    - data: DataFrame containing the data
    - time_column: Name of the column representing time (e.g., Date, Month)
    - value_column: Name of the column representing the values to plot (e.g., Lead Count)
    - title: Title of the plot
    - yaxis_title: Label for the y-axis
    - color_column: Name of the column to color by category (for stacking)
    """
    # Create the stacked bar chart
    fig = px.bar(data, 
                 x=time_column, 
                 y=value_column, 
                 color=color_column,  # Stack bars based on this column (e.g., Zone)
                 title=title, 
                 labels={time_column: 'Time', value_column: yaxis_title},
                 barmode='stack')  # Set the bar mode to stacked
    
    # Update layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title=yaxis_title,
        height=600,
        width=1000,
        template='plotly_dark',  # Suitable for black background
        legend_title=color_column
    )
    
    # Return the figure
    return fig

#%%Load data and perform basic cleaning
lead_univ = load_data()['x']
lead_data_opp = load_data()['y']

# Replace '-'& 'NA' with np.nan
lead_univ = lead_univ.replace('-', np.nan)
lead_univ = lead_univ.replace('N/A', np.nan)
lead_data_opp = lead_data_opp.replace('-', np.nan)
lead_data_opp = lead_data_opp.replace('N/A', np.nan)

#clean leadscore column
lead_univ['[Lead Score]'] = lead_univ['[Lead Score]'].str.replace('%', '')
lead_univ['[Lead Score]'] = lead_univ['[Lead Score]'].astype('float')
#lead_univ['Opportunity ID'] = lead_univ['Opportunity ID'].replace('-', np.nan)

#Extract Year Month
lead_univ['LeadDate'] = pd.to_datetime(lead_univ['Lead Creation Date'])
lead_univ['Year-Month'] = lead_univ['LeadDate'].dt.to_period('M')
lead_data_opp['LeadDate'] = pd.to_datetime(lead_data_opp['Lead Creation Date'])
lead_data_opp['Year-Month'] = lead_data_opp['LeadDate'].dt.to_period('M') 

#replace old BU names with new names
business_units = ['MR', 'PDO', 'HPM', 'US', 'EC', 'Others', 'CVI', 'CI', 'DXR', 'IGT(Fix+MoS)', 'CT', 'IGTD', 'RI', 'EMR&CM', 'AMD', 'CII']
# Dictionary with replacements
replacements = {
    'MR OEM': 'MR',
    'Precision Diagnosis Other': 'PDO',
    'Hospital Patient Monitoring': 'HPM',
    'Ultrasound': 'US',
    'Emergency Care': 'EC',
    'Health Tech other': 'Others',
    'Cardiovascular Informatics': 'CVI',
    'Clinical Informatics': 'CI',
    'Monitoring Others': 'Others',
    'DXR': 'DXR',
    'Image Guided Therapy Systems': 'IGT(Fix+MoS)',
    'CT AMI': 'CT',
    'Image Guided Therapy Devices': 'IGTD',
    'Radiology Informatics': 'RI',
    'EMR & Care Mgt': 'EMR&CM',
    'Image Guided Therapy Others': 'IGT(Fix+MoS)',
    'Innovation & Services': 'Others',
    '8277': 'Others',
    'Ambulatory Monitoring & Diagnostics': 'AMD',
    'Clinical Integration & Insights': 'CII',
    'EPD Solutions': 'Others'
}
# Replace values using the dictionary
lead_univ['Business Unit'] = lead_univ['Business Unit'].replace(replacements)
lead_data_opp['Business Unit'] = lead_data_opp['Business Unit'].replace(replacements)

#replace old stage names with new names
# Dictionary with replacements
replacements_stage = {
    'Opportunity Stage': 'CAT',
    'Sales Recognized': 'WON',
    'Order Booked': 'WON',
    '-': 'L/NP',
    'Cancelled by Customer': 'L/NP',
    'Develop': 'ACTIVE',
    'Identify': 'ACTIVE',
    'Qualify': 'ACTIVE',
    'Lost': 'ACTIVE',
    'Not Pursuing': 'L/NP',
    'Propose': 'ACTIVE',
    'Develop': 'ACTIVE',
    'Order Promised': 'WON',
    'Pre-Qualification (Master)': 'ACTIVE',
    'Qualify (Master)': 'ACTIVE',
    'Pre Qualified': 'L/NP',
    'Customer Cancelled': 'L/NP'
}
# Replace values using the dictionary
lead_univ['Opportunity Stage'] = lead_univ['Opportunity Stage'].replace(replacements_stage)
lead_data_opp['Opportunity Stage'] = lead_data_opp['Opportunity Stage'].replace(replacements)

#%%Define filters in sidebar panel
# Streamlit app code
st.sidebar.title('Leads Data Analysis')

start_date = st.sidebar.date_input('Start date', datetime.date(2022, 9, 1))
end_date = st.sidebar.date_input('End date', lead_univ['Lead Creation Date'].max())

# Helper function to handle 'All' option in multi-select
def handle_multiselect_old(options, selected):
    # If 'All' is selected and other options are selected, deselect 'All'
    if 'All' in selected and len(selected) > 1:
        selected.remove('All')
    # If nothing else is selected, keep 'All' selected
    elif not selected:
        selected = ['All']
    # If 'All' is selected, return all options but remove 'All' from the displayed selected items
    #return [option for option in selected if option != 'All']
    if selected == ['All']:
        return options
    return selected

# Helper function to handle 'All' option in multi-select
def handle_multiselect(options, selected):
    # If 'All' is selected, return all options
    if 'All' in selected:
        return options
    # Otherwise, return the selected values
    else:
        return selected


# Get all unique zones from lead_univ['Zone']
all_zones = lead_univ['Zone'].unique().tolist()
zone_options = ['All'] + all_zones  # Add 'All' option
# Create sidebar multi-select filter for 'Zone'
zone_filter = st.sidebar.multiselect(
    'Select Zone', 
    options=zone_options, 
    default=['All']  # Default to 'All'
)
# Handle 'All' option
zone_filter = handle_multiselect(all_zones, zone_filter)


# For Business Units
business_units_options = ['All'] + business_units  # Add 'All' option
# Create sidebar multi-select filter for 'Business Unit'
bu_filter = st.sidebar.multiselect(
    'Select BU', 
    options=business_units_options, 
    default=['All']  # Default to 'All'
)

# Handle 'All' option
bu_filter = handle_multiselect(business_units, bu_filter)

# For Campaign Channels
cmpgn_chnnl = ['Event Marketing', 'Website', 'Search (Paid)', 'Landing Page', 'Email Marketing', 
               'Public Relations', 'Outbound Call Sales', 'Event', 'Social Media (Paid)', 
               'Search (Display)', 'Trade show', np.nan]

cmpgn_chnnl_options = ['All'] + cmpgn_chnnl  # Add 'All' option
# Create sidebar multi-select filter for 'Campaign Channel'
chnnl_filter = st.sidebar.multiselect(
    'Select Campaign Channel', 
    options=cmpgn_chnnl_options, 
    default=['All']  # Default to 'All'
)
# Handle 'All' option
chnnl_filter = handle_multiselect(cmpgn_chnnl, chnnl_filter)

# Filter data by date range
filtered_data = lead_univ[(lead_univ['Lead Creation Date'] >= start_date) & (lead_univ['Lead Creation Date'] <= end_date)]
filtered_opp = lead_data_opp[(lead_data_opp['Lead Creation Date'] >= start_date) & (lead_data_opp['Lead Creation Date'] <= end_date)]

# Filter data by Zone, Business Unit, and Campaign Channel
filtered_data = filtered_data[
    (filtered_data['Zone'].isin(zone_filter)) &
    (filtered_data['Business Unit'].isin(bu_filter)) &
    (filtered_data['[Campaign - Channel]'].isin(chnnl_filter))
]

filtered_opp = filtered_opp[
    (filtered_opp['Zone'].isin(zone_filter)) &
    (filtered_opp['Business Unit'].isin(bu_filter)) &
    (filtered_opp['[Campaign - Channel]'].isin(chnnl_filter))
]

#---------------------------------------------------------------#
#------------------ P   L   O   T   S   ------------------------#
#---------------------------------------------------------------#

#%% Subheader for Q1
st.subheader('1. Which zones generate the most leads?', divider='rainbow')

# Count leads and conversions (opportunities) by zone
zones_counts_ts = filtered_data.groupby(['Year-Month', 'Zone']).agg(
    total_leads=('Lead ID', 'count'),
    total_mqls=('YTD MQLs', lambda x: x.sum()),
).reset_index()
zones_counts_ts['Year-Month'] = zones_counts_ts['Year-Month'].astype(str)

# Call the function to create the stacked bar chart
fig_zone_lead = create_stacked_bar_ts(
    data=zones_counts_ts,
    time_column='Year-Month',
    value_column='total_leads',
    title='Leads Generated Over Time by Zone',
    yaxis_title='Number of Leads',
    color_column='Zone'  # Stacked bars by Zone
)

fig_zone_mql = create_stacked_bar_ts(
    data=zones_counts_ts,
    time_column='Year-Month',
    value_column='total_mqls',
    title='MQLs Generated Over Time by Zone',
    yaxis_title='Number of MQLs',
    color_column='Zone'  # Stacked bars by Zone
)

# Count leads and conversions (opportunities) by BU
bu_counts_ts = filtered_data.groupby(['Year-Month', 'Business Unit']).agg(
    total_leads=('Lead ID', 'count'),
    total_mqls=('YTD MQLs', lambda x: x.sum()),
).reset_index()
bu_counts_ts['Year-Month'] = bu_counts_ts['Year-Month'].astype(str)

# Call the function to create the stacked bar chart
fig_bu_lead = create_stacked_bar_ts(
    data=bu_counts_ts,
    time_column='Year-Month',
    value_column='total_leads',
    title='Leads Generated Over Time by Business Unit',
    yaxis_title='Number of Leads',
    color_column='Business Unit'  # Stacked bars by Zone
)

fig_bu_mql = create_stacked_bar_ts(
    data=bu_counts_ts,
    time_column='Year-Month',
    value_column='total_mqls',
    title='MQLs Generated Over Time by Zone',
    yaxis_title='Number of MQLs',
    color_column='Business Unit'  # Stacked bars by Zone
)

# Step 8: Create a bar chart to visualize the mean time difference across business units
on_mqls = st.toggle("Show MQLs", key="q1")

if on_mqls:
    st.plotly_chart(fig_zone_mql, key='1a')
    st.plotly_chart(fig_bu_mql, key='1b')
else:
    st.plotly_chart(fig_zone_lead, key='1c')
    st.plotly_chart(fig_bu_lead, key='1d')

 
   
#%% Subheader for Q2
st.subheader('2. What are the most common lead sources?', divider='rainbow')
# Create pie charts for 'Lead Source' and 'Lead Source Original'
st.markdown("<h3 style='font-size:18px;'>Source of Leads</h3>", unsafe_allow_html=True)

lead_source_counts = filtered_data['Lead Source'].value_counts().reset_index()
lead_source_counts.columns = ['Lead Source', 'Count']

# Create sunburst plot for original category
lead_catg_counts = filtered_data['Lead_Source_Catg'].value_counts().reset_index()
lead_catg_counts.columns = ['Lead_Category', 'Count']

# Add a 'Parent' column for the hierarchy
lead_catg_counts['Parent'] = lead_catg_counts['Lead_Category'].apply(lambda x: 'Lead Source' if x == 'Event' else 'Digital')

# Manually append the parent 'Digital' and 'Lead Source' rows to complete the hierarchy
lead_catg_counts = pd.concat([
    lead_catg_counts,
    pd.DataFrame({'Lead_Category': 'Digital', 'Count': lead_catg_counts[lead_catg_counts['Parent'] == 'Digital']['Count'].sum(), 'Parent': 'Lead Source'}, index=[0])
]).reset_index(drop=True)

lead_catg_counts = pd.concat([
    lead_catg_counts,
    pd.DataFrame({'Lead_Category': 'Lead Source', 'Count': lead_catg_counts[lead_catg_counts['Parent'] == 'Lead Source']['Count'].sum(), 'Parent': ''}, index=[0])
]).reset_index(drop=True)

# Prepare the Sunburst Chart with both percentage and absolute values in labels
fig_src = go.Figure(go.Sunburst(
    labels=lead_catg_counts['Lead_Category'],
    parents=lead_catg_counts['Parent'],
    values=lead_catg_counts['Count'],
    branchvalues="total",
    hoverinfo="label+percent parent+value",  # Show label, percent of parent, and absolute value on hover
    textinfo='label+value+percent entry',  # Display label, value, and percentage on the chart itself
    insidetextorientation='radial'  # Orientation of text inside the segments
))

# Update layout for better display
fig_src.update_layout(
    margin=dict(t=0, l=0, r=0, b=0),
    title_text="Lead Source Hierarchy"
)

# Display the chart
st.plotly_chart(fig_src, key='2a')

# Group data by Zone and Date to get the lead count over time
lead_src_counts_ts = filtered_data.groupby(['Year-Month', 'Lead_Source_Catg']).size().reset_index(name='Lead Count')
lead_src_counts_ts['Year-Month'] = lead_src_counts_ts['Year-Month'].astype(str)

# Call the function to create the stacked bar chart
fig_src_ts = create_stacked_bar_ts(
    data=lead_src_counts_ts,
    time_column='Year-Month',
    value_column='Lead Count',
    title='Leads Generated Over Time by Lead Source',
    yaxis_title='Number of Leads',
    color_column='Lead_Source_Catg'  # Stacked bars by Zone
)
st.plotly_chart(fig_src_ts, key='2b')

#%% Subheader for Q3
st.subheader('3. What proportion of leads are getting converted to opportunities?', divider='rainbow')

##---------------- ZONE --------------------##
# Count leads and conversions (opportunities) by zone
conversion_data = filtered_data.groupby('Zone').agg(
    total_leads=('Lead ID', 'count'),
    leads_to_mql=('YTD MQLs', lambda x: x.sum()),
    mql_to_opp=('Opportunity ID', lambda x: x.notna().sum())
).reset_index()

# Calculate proportion of converted leads
conversion_data['conv_leads_mql'] = conversion_data['leads_to_mql'] / conversion_data['total_leads']
conversion_data['conv_mql_opp'] = conversion_data['mql_to_opp'] / conversion_data['leads_to_mql']

# Sort the data by total leads in descending order
conversion_data = conversion_data.sort_values(by='total_leads', ascending=False)

# Melt data to get it in long format for stacked bar plot
melted_data = pd.melt(conversion_data, id_vars='Zone', value_vars=['mql_to_opp', 'leads_to_mql', 'total_leads'],
                      var_name='Lead Status', value_name='Count')

# Adjust for stacked bar chart where 'converted_leads' needs to be shown as converted portion of the total
melted_data['Lead Status'] = melted_data['Lead Status'].map({
    'mql_to_opp': 'Opportunities',
    'leads_to_mql': 'MQLs',
    'total_leads': 'Leads'
})

# Define the category order for 'Lead Status' so that 'Opportunities' comes last
category_order = ['Leads', 'MQLs', 'Opportunities']
# Create the overlay bar chart using plotly
fig_conv_zone = px.bar(melted_data,
             x='Zone', 
             y='Count', 
             color='Lead Status', 
             text='Count', 
             title='Proportion of Leads Converted to Opportunities by Zone', 
             labels={'Zone': 'Zone', 'Count': 'Number of Leads'},
             color_discrete_map={'Opportunities': '#32a852', 'MQLs': '#ffdd57', 'Leads': '#ff6f61'},
             category_orders={'Lead Status': category_order},  # Ensure correct plotting order
             barmode='overlay')  # Stack the bars to show proportion

# Update layout
fig_conv_zone.update_layout(xaxis_title='Zone', 
                  yaxis_title='Number of Leads', 
                  legend_title='Lead Status',
                  height=600,
                  font_color='white',    # White font for labels and text
                  xaxis={'categoryorder':'total descending'},  # Sort the x-axis based on total_leads
                  )

# Update text visibility and position
# Set bars to be fully opaque
fig_conv_zone.update_traces(
    opacity=1,  # Set opacity to 1 (fully opaque)
    texttemplate='%{text:.0f}',  # Display the count without decimals
    textposition='outside'  # Position text outside the bars
)

col1, col2 = st.columns([2,1], gap="small")
with col1:
    # Display the plot in the Streamlit app
    st.plotly_chart(fig_conv_zone, key='3a')
    #plotly.offline.plot(fig)

with col2:
    # Show conversion rates for each zone as a table
    st.markdown("<h3 style='font-size:18px;'>Conversion Rates by Zone</h3>", unsafe_allow_html=True)
    # Show conversion rates as a percentage
    conversion_data['conv_leads_mql'] = conversion_data['conv_leads_mql'].map("{:.2%}".format)
    conversion_data['conv_mql_opp'] = conversion_data['conv_mql_opp'].map("{:.2%}".format)
    st.dataframe(conversion_data[['Zone', 'conv_leads_mql', 'conv_mql_opp']].rename(columns={'conv_leads_mql':'Leads to MQL(%)', 'conv_mql_opp':'MQLs to Opp.ty(%)'}), hide_index=True)
    
##---------------- BUSINESS UNIT --------------------##
# Count leads and conversions (opportunities) by zone
conversion_data = filtered_data.groupby('Business Unit').agg(
    total_leads=('Lead ID', 'count'),
    leads_to_mql=('YTD MQLs', lambda x: x.sum()),
    mql_to_opp=('Opportunity ID', lambda x: x.notna().sum())
).reset_index()

# Calculate proportion of converted leads
conversion_data['conv_leads_mql'] = conversion_data['leads_to_mql'] / conversion_data['total_leads']
conversion_data['conv_mql_opp'] = conversion_data['mql_to_opp'] / conversion_data['leads_to_mql']

# Sort the data by total leads in descending order
conversion_data = conversion_data.sort_values(by='total_leads', ascending=False)

# Melt data to get it in long format for stacked bar plot
melted_data = pd.melt(conversion_data, id_vars='Business Unit', value_vars=['mql_to_opp', 'leads_to_mql', 'total_leads'],
                      var_name='Lead Status', value_name='Count')

# Adjust for stacked bar chart where 'converted_leads' needs to be shown as converted portion of the total
melted_data['Lead Status'] = melted_data['Lead Status'].map({
    'mql_to_opp': 'Opportunities',
    'leads_to_mql': 'MQLs',
    'total_leads': 'Leads'
})

# Define the category order for 'Lead Status' so that 'Opportunities' comes last
category_order = ['Leads', 'MQLs', 'Opportunities']
# Create the overlay bar chart using plotly
fig_conv_bu = px.bar(melted_data,
             x='Business Unit', 
             y='Count', 
             color='Lead Status', 
             text='Count', 
             title='Proportion of Leads Converted to Opportunities by Business Unit', 
             labels={'Business Unit': 'Business Unit', 'Count': 'Number of Leads'},
             color_discrete_map={'Opportunities': '#32a852', 'MQLs': '#ffdd57', 'Leads': '#ff6f61'},
             category_orders={'Lead Status': category_order},  # Ensure correct plotting order
             barmode='overlay')  # Stack the bars to show proportion

# Update layout
fig_conv_bu.update_layout(xaxis_title='Business Unit', 
                  yaxis_title='Number of Leads', 
                  legend_title='Lead Status',
                  height=600,
                  font_color='white',    # White font for labels and text
                  xaxis={'categoryorder':'total descending'},  # Sort the x-axis based on total_leads
                  )

# Update text visibility and position
# Set bars to be fully opaque
fig_conv_bu.update_traces(
    opacity=1,  # Set opacity to 1 (fully opaque)
    texttemplate='%{text:.0f}',  # Display the count without decimals
    textposition='outside'  # Position text outside the bars
)

col1, col2 = st.columns([2,1], gap="small")
with col1:
    # Display the plot in the Streamlit app
    st.plotly_chart(fig_conv_bu, key='3b')
    #plotly.offline.plot(fig)

with col2:
    # Show conversion rates for each zone as a table
    st.markdown("<h3 style='font-size:18px;'>Conversion Rates by Business Unit</h3>", unsafe_allow_html=True)
    # Show conversion rates as a percentage
    conversion_data['conv_leads_mql'] = conversion_data['conv_leads_mql'].map("{:.2%}".format)
    conversion_data['conv_mql_opp'] = conversion_data['conv_mql_opp'].map("{:.2%}".format)
    st.dataframe(conversion_data[['Business Unit', 'conv_leads_mql', 'conv_mql_opp']].rename(columns={'conv_leads_mql':'Leads to MQL(%)', 'conv_mql_opp':'MQLs to Opp.ty(%)'}), hide_index=True)



#%% Subheader for Q4
st.subheader('4. What is the average time to convert a Marketing Qualified Lead (MQL) to an opportunity?', divider='rainbow')

# Convert end_date to 'Year-Month' format (YYYY-MM)
end_year = end_date.year
end_month = end_date.month

# Step 1: Map abbreviated month names to their corresponding numbers
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# Step 2: Convert the 'Month' column in filtered_data to numeric months using the mapping
filtered_data['Month_Num'] = filtered_data['Month'].map(month_mapping)

# Helper function to calculate the previous year date
def get_previous_year(year):
    return year - 1

# Filter data for the current year and previous year (YTD)
filtered_data['MQL to Opp. Lead Velocity (Days)'] = filtered_data['MQL to Opp. Lead Velocity (Days)'].replace('-', np.nan).astype(float)

# Filter for the current year (up to selected month)
current_year_data = filtered_data[(filtered_data['Year'] == end_year) & (filtered_data['Month_Num'] <= end_month)]

# Filter for the previous year (up to the same month)
previous_year_data = filtered_data[(filtered_data['Year'] == get_previous_year(end_year)) & (filtered_data['Month_Num'] <= end_month)]

# Step 2: Group by 'Campaign Channel' and calculate the YTD average for both years
current_year_agg = current_year_data.groupby('[Campaign - Channel]').agg(
    avg_conversion_time=('MQL to Opp. Lead Velocity (Days)', 'mean')
).reset_index()

previous_year_agg = previous_year_data.groupby('[Campaign - Channel]').agg(
    avg_conversion_time=('MQL to Opp. Lead Velocity (Days)', 'mean')
).reset_index()

# Step 3: Merge current and previous year data on 'Campaign Channel'
conversion_data = pd.merge(
    previous_year_agg.rename(columns={'avg_conversion_time': 'prev_year_avg_conversion_time'}),
    current_year_agg.rename(columns={'avg_conversion_time': 'current_year_avg_conversion_time'}),
    on='[Campaign - Channel]',
    how='inner'
)

end_month_label = end_date.strftime('%Y-%m')
previous_year_label = f"{get_previous_year(end_year)}-{end_month:02d}"

# Step 4: Create the candlestick chart
candlestick = go.Figure()

# Add candlestick trace
candlestick.add_trace(go.Candlestick(
    x=conversion_data['[Campaign - Channel]'],  # x-axis is the Campaign Channel
    open=conversion_data['prev_year_avg_conversion_time'],  # previous year YTD value
    close=conversion_data['current_year_avg_conversion_time'],  # current year YTD value
    low=conversion_data[['prev_year_avg_conversion_time', 'current_year_avg_conversion_time']].min(axis=1),  # min between current and previous
    high=conversion_data[['prev_year_avg_conversion_time', 'current_year_avg_conversion_time']].max(axis=1),  # max between current and previous
    increasing_line_color='red',  # green for decreasing avg conversion time
    decreasing_line_color='green',  # red for increasing avg conversion time
    showlegend=False
))

# Step 5: Add text annotations for the previous year and current year
for i, row in conversion_data.iterrows():
    # Add text label for previous year value
    candlestick.add_annotation(
        x=row['[Campaign - Channel]'],
        y=row['prev_year_avg_conversion_time'],
        #text=f"{previous_year_label}: {row['prev_year_avg_conversion_time']:.1f} d",
        text=f"From: {row['prev_year_avg_conversion_time']:.1f} days ({previous_year_label})",
        showarrow=False,
        yshift=10,
        font=dict(color="white", size=10)
    )
    
    # Add text label for current year value
    candlestick.add_annotation(
        x=row['[Campaign - Channel]'],
        y=row['current_year_avg_conversion_time'],
        #text=f"{end_month_label}: {row['current_year_avg_conversion_time']:.1f} d",
        text=f"To: {row['current_year_avg_conversion_time']:.1f} days ({end_month_label})",
        showarrow=False,
        yshift=-15,
        font=dict(color="white", size=10)
    )
    
    # Step 6: Add arrows to show increase or decrease
    if row['current_year_avg_conversion_time'] <= row['prev_year_avg_conversion_time']:
        arrow_color = 'green'
        arrow_direction = -20
        arrow_side= 'start'
    else:
        arrow_color = 'red'
        arrow_direction = 20
        arrow_side= 'end'
    
    # Add arrow annotation
    candlestick.add_annotation(
        x=row['[Campaign - Channel]'],
        y=(row[['prev_year_avg_conversion_time', 'current_year_avg_conversion_time']].mean()),  # arrow between previous and current year
        axref='x',
        ay=row['prev_year_avg_conversion_time'],
        ax=row['[Campaign - Channel]'],
        #yshift=arrow_direction,
        arrowhead=3, #3
        arrowsize=1.5,
        arrowwidth=2,
        arrowside=arrow_side,
        arrowcolor=arrow_color
    )

# Customize the layout
candlestick.update_layout(
    title="YTD Change in Average MQL to Opportunity Conversion Time by Campaign Channel",
    xaxis_title="Campaign Channel",
    yaxis_title="Avg Conversion Time (Days)",
    xaxis_rangeslider_visible=False,
    margin=dict(t=60, b=100, l=40, r=40),
    height=600,
    width=800
)

# Step 4: Display the plots
st.plotly_chart(candlestick, key='4a')

#%% Subheader for Q5

st.subheader('5. Which campaigns result in the highest lead conversion rates?', divider='rainbow')

# Group by 'Campaign Name' and calculate total leads and converted leads
campaign_conversion_data = filtered_data.groupby('Campaign Name').agg(
    total_leads=('Lead ID', 'count'),
    converted_leads=('Opportunity ID', lambda x: x.notna().sum())
).reset_index()

# Calculate conversion rate
campaign_conversion_data['conversion_rate'] = round((campaign_conversion_data['converted_leads'] / campaign_conversion_data['total_leads']) * 100, 2)
# Filter rows where conversion rate > 0
campaign_conversion_data = campaign_conversion_data[campaign_conversion_data['conversion_rate'] > 0]

# Sort the data by total_leads in descending order
campaign_conversion_data = campaign_conversion_data.sort_values(by='conversion_rate', ascending=False)

# Create a bar chart to display the conversion rate by campaign
fig_campaign_conversion = px.bar(
    campaign_conversion_data, 
    x='conversion_rate', 
    y='Campaign Name', 
    text='conversion_rate', 
    title='Lead Conversion Rates by Campaign',
    labels={'Campaign Name': 'Campaign Name', 'conversion_rate': 'Conversion Rate (%)'},
    color='total_leads',
    color_continuous_scale=px.colors.sequential.Viridis,
    orientation='h')

# Update layout for better display
fig_campaign_conversion.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_campaign_conversion.update_layout(
    xaxis_title='Conversion Rate (%)', 
    yaxis_title='Campaign Name', 
    yaxis={'categoryorder':'total ascending'},  # Sort the y-axis based on total_leads
    height=900,
    width=1000,
    showlegend=False
)
#plotly.offline.plot(fig_campaign_conversion)

# Sort by total leads in descending order
campaign_conversion_data = campaign_conversion_data.sort_values(by='total_leads', ascending=False)

# Compute cumulative percentage of total leads
campaign_conversion_data['cumulative_leads'] = campaign_conversion_data['total_leads'].cumsum()
campaign_conversion_data['cumulative_leads_percentage'] = (campaign_conversion_data['cumulative_leads'] / campaign_conversion_data['total_leads'].sum()) * 100

on_campaign = st.toggle("Pareto View", key='q5')

if on_campaign:
    # User input for percentage threshold (e.g., 80%)
    percentage_threshold = st.selectbox('Select percentage of total leads to show:', [50, 60, 70, 80, 90, 100], index=3)

    # Filter the campaigns contributing to the selected percentage of total leads
    filtered_campaigns = campaign_conversion_data[campaign_conversion_data['cumulative_leads_percentage'] <= percentage_threshold]

    # Create a Pareto chart to display the conversion rate by campaign
    fig_campaign_pareto = px.bar(
        filtered_campaigns, 
        x='conversion_rate', 
        y='Campaign Name', 
        text='conversion_rate', 
        title=f'Lead Conversion Rates by Campaign (Top {percentage_threshold}% of Total Leads)',
        labels={'Campaign Name': 'Campaign Name', 'conversion_rate': 'Conversion Rate (%)'},
        color='total_leads',
        color_continuous_scale=px.colors.sequential.Viridis,
        orientation='h')

    # Update layout for better display
    fig_campaign_pareto.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_campaign_pareto.update_layout(
        xaxis_title='Conversion Rate (%)', 
        yaxis_title='Campaign Name', 
        yaxis={'categoryorder':'total ascending'},  # Sort the y-axis based on total_leads
        height=900,
        width=1000,
        showlegend=False
    )
    # Display the plot in the Streamlit app
    st.plotly_chart(fig_campaign_pareto, key='6a')
else:
    st.plotly_chart(fig_campaign_conversion, key='6b')

# Show the data for reference
st.markdown("<h3 style='font-size:18px;'>Campaign Conversion Data</h3>", unsafe_allow_html=True)
st.dataframe(campaign_conversion_data[['Campaign Name', 'total_leads', 'converted_leads', 'conversion_rate']].rename(columns={
    'conversion_rate': 'Conversion Rate (%)'
}), hide_index=True)

#%% Subheader for Q6
st.subheader('6. What is the order value associated with leads from specific campaigns?', divider='rainbow')

# Assuming filtered_opp is your DataFrame

# Step 1: Group by 'Campaign Name' and sum the 'Opportunity Amount Updated' to get total order value
campaign_ov_data = filtered_opp.groupby('Campaign Name').agg(
    total_order_value=('Opportunity Amount Updated', 'sum')
).reset_index()

# Filter rows where total order value > 0
campaign_ov_data = campaign_ov_data[campaign_ov_data['total_order_value'] > 0]

# Sort the data by total order value in descending order
campaign_ov_data = campaign_ov_data.sort_values(by='total_order_value', ascending=False)

# Step 2: Calculate cumulative contribution for Pareto principle
campaign_ov_data['Cumulative Contribution'] = campaign_ov_data['total_order_value'].cumsum() / campaign_ov_data['total_order_value'].sum() * 100

# Step 3: Filter for the top 80% contributors (Pareto view)
pareto_data = campaign_ov_data[campaign_ov_data['Cumulative Contribution'] <= 80]

# Step 4: Group the remaining categories as "Others" for the Pareto view
remaining_categories = campaign_ov_data[campaign_ov_data['Cumulative Contribution'] > 80]
others = pd.DataFrame({
    'Campaign Name': ['Others'],
    'total_order_value': [remaining_categories['total_order_value'].sum()],
    'Cumulative Contribution': [100.0]  # Since 'Others' will sum up to 100%
})

# Combine the top 80% campaigns with the 'Others' category
#pareto_with_others = pd.concat([pareto_data, others])

# Step 5: Create the figure object for both normal and Pareto views

# Step 4: Create a bar chart to visualize the mean time difference across business units
on = st.toggle("Pareto View", key='q6')

if on:
    fig_pareto_ov = px.bar(
    pareto_data, 
    x='total_order_value', 
    y='Campaign Name', 
    text='total_order_value', 
    title='Total Order Opportunity Value by Campaign (Pareto)',
    labels={'Campaign Name': 'Campaign Name', 'total_order_value': 'Total Order Value (€)'},
    orientation='h'
    )

    # Update layout for better display
    fig_pareto_ov.update_traces(texttemplate='%{text:.3s}', textposition='inside')
    fig_pareto_ov.update_layout(
        yaxis_title='Campaign Name', 
        xaxis_title='Total Order Value (€)', 
        yaxis={'categoryorder':'total ascending'},  # Sort the x-axis based on total order value
        height=900,
        width=1000,
        showlegend=False
    )
    st.plotly_chart(fig_pareto_ov, key='6c')
else:
    fig_campaign_ov = px.bar(
    campaign_ov_data,
    x='total_order_value', 
    y='Campaign Name', 
    text='total_order_value', 
    title='Total Order Opportunity Value by Campaign',
    labels={'Campaign Name': 'Campaign Name', 'total_order_value': 'Total Order Value (€)'},
    color='total_order_value',
    color_continuous_scale=px.colors.sequential.Blues,
    orientation='h')

    # Update layout for better display
    fig_campaign_ov.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_campaign_ov.update_layout(
        yaxis_title='Campaign Name', 
        xaxis_title='Total Order Value (€)', 
        yaxis={'categoryorder':'total ascending'},  # Sort the x-axis based on total order value
        height=900,
        width=1000,
        showlegend=False
    )

    st.plotly_chart(fig_campaign_ov, key='6d')

#%%Subheader for Q7
st.subheader('7. Which business units generate the highest quality leads?', divider='rainbow')
# Group the data by 'Business Unit' and 'Lead Qualification Level' to count the number of leads in each category
bu_lead_quality_data = filtered_data.groupby(['Business Unit', 'Lead Qualification Level']).size().reset_index(name='Lead Count')

# Create a stacked bar chart to show lead quality across business units
fig_bu_lead_quality = px.bar(
    bu_lead_quality_data, 
    x='Business Unit', 
    y='Lead Count', 
    color='Lead Qualification Level',
    title='Lead Quality by Business Unit',
    labels={'Business Unit': 'Business Unit', 'Lead Count': 'Number of Leads', 'Lead Qualification Level': 'Lead Quality'},
    barmode='stack',  # Stacked bars to show distribution
    color_discrete_map={'Hot': 'red', 'Warm': 'orange', 'Cold': 'blue', 'Disqualified': 'grey'}  # Custom colors for lead quality
)

# Update layout for better visualization
fig_bu_lead_quality.update_layout(
    xaxis_title='Business Unit', 
    yaxis_title='Number of Leads',
    xaxis={'categoryorder':'total descending'},  # Sort the x-axis based on total order value
    height=600
)

col1, col2 = st.columns([1,1], gap="small")
with col1:
    # Display the plot in the Streamlit app
    st.plotly_chart(fig_bu_lead_quality, use_container_width=True, key='7a')
with col2:
    # Show the grouped data for reference
    st.markdown("<h3 style='font-size:18px;'>Business Unit Lead Quality Data</h3>", unsafe_allow_html=True)
    if len(bu_lead_quality_data)>0:
        # Pivot the data for cross-sectional view
        pivot_bu_lead_quality = bu_lead_quality_data.pivot(index='Business Unit', columns='Lead Qualification Level', values='Lead Count')
        pivot_bu_lead_quality = pivot_bu_lead_quality[['Hot', 'Warm', 'Cold', 'Disqualified']]  # Reordering the columns
        
        # Fill NaN with 0 for display purposes
        pivot_bu_lead_quality = pivot_bu_lead_quality.fillna(0).astype(int)
        st.dataframe(pivot_bu_lead_quality)


# Create a contingency table for 'Business Unit' and 'Lead Qualification Level'
contingency_table_bu = pd.crosstab(filtered_data['Business Unit'], filtered_data['Lead Qualification Level'])

# 1. Perform Chi-Square test to assess the relationship between 'Business Unit' and 'Lead Qualification Level'
chi2_stat_bu, p_val_bu, dof_bu, expected_bu = stats.chi2_contingency(contingency_table_bu)

# 2. Calculate standardized residuals (Observed - Expected / sqrt(Expected))
observed_bu = contingency_table_bu.values
residuals_bu = (observed_bu - expected_bu) / np.sqrt(expected_bu)
    
on_chisq_q7 = st.toggle("View Statistical Test of Association", key='chiq7')

if on_chisq_q7:
    # 3. Display the result of the Chi-Square test
    st.markdown("<h3 style='font-size:18px;'>Chi-Square Test of Association between Business Unit vs Lead Qualification Level</h3>", unsafe_allow_html=True)
    st.write(f"Chi-Square Statistic: {chi2_stat_bu:.2f}, Degrees of Freedom: {dof_bu}, P-Value: {p_val_bu:.4f}")
    
    # Conclusion based on the P-value
    if p_val_bu < 0.05:
        st.write("Conclusion: There is a significant relationship between Business Unit and Lead Qualification Level.")
    else:
        st.write("Conclusion: There is NO significant relationship between Business Unit and Lead Qualification Level.")
    
    # 4. Create a DataFrame for residuals and plot the heatmap to visualize significant relationships
    residuals_bu_df = pd.DataFrame(residuals_bu, index=contingency_table_bu.index, columns=contingency_table_bu.columns)
    
    # Flatten the residuals dataframe for plotting
    residuals_bu_flat = residuals_bu_df.reset_index().melt(id_vars='Business Unit', var_name='Lead Qualification Level', value_name='Residual')
    
    # 5. Plot residuals as a heatmap using Plotly
    fig_residuals_bu = px.imshow(residuals_bu_df, 
                                 title="Standardized Residuals: Business Unit vs Lead Qualification Level",
                                 labels=dict(color="Standardized Residual"),
                                 aspect="auto")
    
    # Display heatmap
    st.plotly_chart(fig_residuals_bu, key='7b')
    
    st.write('Positive residuals indicate a higher observed count than expected, while negative residuals indicate a lower observed count than expected.')


#%%# Subheader for Q8
st.subheader('8. What are the Main Reasons for Lead Rejection?', divider='rainbow')

# List of values to exclude
excluded_values = ['Transferred To Channel Partner', 'Does Not Apply', 'None']

# Step 1: Add a switch (checkbox) to toggle exclusion of the specified values
exclude_values = st.checkbox("Exclude extra rejection reasons")

# Step 2: Apply the filter if the switch is toggled
if exclude_values:
    filtered_data_plot = filtered_data[~filtered_data['Reason for Rejection'].isin(excluded_values)]
else:
    filtered_data_plot = filtered_data

# Step 3: Group by 'Reason for Rejection' and count occurrences
reason_rejection_data = filtered_data_plot['Reason for Rejection'].value_counts().reset_index()
reason_rejection_data.columns = ['Reason for Rejection', 'Lead Count']

# Step 4: Plot bar chart for reasons of rejection
fig_rejection_reasons = px.bar(
    reason_rejection_data,
    x='Reason for Rejection',
    y='Lead Count',
    text='Lead Count',
    title='Reasons for Lead Rejection',
    labels={'Reason for Rejection': 'Rejection Reason', 'Lead Count': 'Number of Leads'}
)

# Step 5: Update layout for better display
fig_rejection_reasons.update_traces(textposition='outside')
fig_rejection_reasons.update_layout(
    xaxis_title='Reason for Rejection',
    yaxis_title='Number of Leads',
    xaxis_tickangle=-45,
    height=600
)

# Step 6: Display the plot
st.plotly_chart(fig_rejection_reasons, key='8a')

#%% Subheader for Q9
st.subheader('9. How effective are different lead actions in converting leads to opportunities?', divider='rainbow')
# 1. Group by 'Lead Action' and 'Opportunity Stage' to get counts
lead_action_vs_stage = filtered_data.groupby(['Lead Action', 'Opportunity Stage']).size().reset_index(name='Count')

# 2. Plotting the stacked bar chart
fig_lead_action_stage = px.bar(
    lead_action_vs_stage,
    x='Opportunity Stage',
    y='Count',
    color='Lead Action',
    title='Effectiveness of Different Lead Actions in Converting to Opportunity Stages',
    labels={'Opportunity Stage': 'Opportunity Stage', 'Count': 'Lead Action', 'Lead Action': 'Lead Action'},
    barmode='stack'
)

# Adjust layout
fig_lead_action_stage.update_layout(
    xaxis_title='Opportunity Stage',
    yaxis_title='Number of Leads',
    height=700
)

# Display the plot
st.plotly_chart(fig_lead_action_stage, key='9a')

on_chisq_q9 = st.toggle("View Statistical Test of Association", key='chiq9')

if on_chisq_q9:
    # 3. Perform Chi-Square test to assess the relationship between Lead Action and Opportunity Stage
    # Create contingency table (cross-tabulation)
    contingency_table = pd.crosstab(filtered_data['Lead Action'], filtered_data['Opportunity Stage'])
    
    # Perform Chi-Square Test
    chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)
    
    # 4. Compute standardized residuals (Observed - Expected / sqrt(Expected))
    observed = contingency_table.values
    expected = ex
    residuals = (observed - expected) / np.sqrt(expected)
    
    # Display the result of the Chi-Square test
    st.markdown("<h3 style='font-size:18px;'>Chi-Square Test of Association between Lead Action vs Opportunity Stage</h3>", unsafe_allow_html=True)
    st.write(f"Chi-Square Statistic: {chi2_stat:.2f}, Degrees of Freedom: {dof}, P-Value: {p_val:.4f}")
    
    # Conclusion based on the P-value
    if p_val < 0.05:
        st.write("Conclusion: There is a significant relationship between Lead Action and Opportunity Stage.")
    else:
        st.write("Conclusion: There is no significant relationship between Lead Action and Opportunity Stage.")
    
    # 5. Display residuals as a heatmap to identify specific relationships
    residuals_df = pd.DataFrame(residuals, index=contingency_table.index, columns=contingency_table.columns)
    
    # Flatten the residuals dataframe for plotting
    residuals_flat = residuals_df.reset_index().melt(id_vars='Lead Action', var_name='Opportunity Stage', value_name='Residual')
    
    # Plot residuals as heatmap using Plotly
    fig_residuals = px.imshow(residuals_df, 
                              title="Standardized Residuals: Lead Action vs. Opportunity Stage",
                              labels=dict(color="Standardized Residual"),
                              aspect="auto")
    
    # Display heatmap
    st.plotly_chart(fig_residuals, key='9b')
    
    st.write('Positive residuals indicate a higher observed count than expected, while negative residuals indicate a lower observed count than expected.')

#%% Subheader for Q10
st.subheader('10. What are the popular products of interest among leads?', divider='rainbow')

# Step 1: Count occurrences of each product
product_counts = filtered_data['Product of Interest'].value_counts().reset_index()
product_counts.columns = ['Product of Interest', 'Lead Count']

# Step 2: Sort by Lead Count in descending order
product_counts = product_counts.sort_values(by='Lead Count', ascending=False)

# Step 3: Calculate cumulative percentage for Pareto calculation
product_counts['cumulative_lead_count'] = product_counts['Lead Count'].cumsum()
product_counts['cumulative_percent'] = (product_counts['cumulative_lead_count'] / product_counts['Lead Count'].sum()) * 100

# Step 4: Add toggle for Pareto functionality (show products contributing to 80% of lead count)
pareto_prod = st.toggle("Pareto View", key="q10")

# Step 5: Apply Pareto filter if toggle is on
if pareto_prod:
    product_counts = product_counts[product_counts['cumulative_percent'] <= 80]

# Step 6: Plot the bar chart with horizontal orientation
fig_product_interest = px.bar(product_counts, 
                              y='Product of Interest',  # Products on y-axis
                              x='Lead Count',  # Lead Count on x-axis
                              text='Lead Count', 
                              title='Trending Products Among Leads',
                              labels={'Product of Interest': 'Product', 'Lead Count': 'Number of Leads'},
                              orientation='h')  # Set horizontal orientation

# Step 7: Adjust layout to show bars in descending order
fig_product_interest.update_traces(textposition='outside')
fig_product_interest.update_layout(xaxis_title='Number of Leads', 
                                   yaxis_title='Product of Interest',
                                   yaxis={'categoryorder': 'total ascending'},  # Sort y-axis by product popularity
                                   height=900)

# Step 8: Display the plot in the Streamlit app
st.plotly_chart(fig_product_interest, key='10a')

##----------------------------------------------##
##-------------- M   Q   L  --------------------##
##----------------------------------------------##

#%%11. Which lead sources have the highest conversion rates?
st.subheader('11. Which lead sources have the highest conversion rates?', divider='rainbow')

# Convert end_date to 'Year-Month' format (YYYY-MM)
end_year = end_date.year
end_month = end_date.month

# Step 1: Map abbreviated month names to their corresponding numbers
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# Step 2: Convert the 'Month' column in filtered_data to numeric months using the mapping
filtered_data['Month_Num'] = filtered_data['Month'].map(month_mapping)

# Function to calculate previous year
def get_previous_year(year):
    return year - 1

# Step 3: Filter the data for the current year and previous year, up to the selected end month
current_year_data = filtered_data[(filtered_data['Year'] == end_year) & (filtered_data['Month_Num'] <= end_month)]
previous_year_data = filtered_data[(filtered_data['Year'] == get_previous_year(end_year)) & (filtered_data['Month_Num'] <= end_month)]

# Group by Lead Source and calculate cumulative metrics for YTD comparison
current_year_agg = current_year_data.groupby('Lead Source Original').agg(
    total_mqls=('YTD MQLs', 'sum'),
    converted_leads=('Opportunity ID', lambda x: x.notna().sum())
).reset_index()

previous_year_agg = previous_year_data.groupby('Lead Source Original').agg(
    total_mqls=('YTD MQLs', 'sum'),
    converted_leads=('Opportunity ID', lambda x: x.notna().sum())
).reset_index()

# Add the 'Year' column for differentiation between current and previous year
current_year_agg['Year'] = end_year
previous_year_agg['Year'] = get_previous_year(end_year)

# Calculate conversion rates
current_year_agg['conversion_rate'] = round((current_year_agg['converted_leads'] / current_year_agg['total_mqls']) * 100, 0)
previous_year_agg['conversion_rate'] = round((previous_year_agg['converted_leads'] / previous_year_agg['total_mqls']) * 100, 0)

# Combine both datasets into one for plotting
combined_data = pd.concat([current_year_agg, previous_year_agg])
combined_data['Year'] = combined_data['Year'].astype(str)

# Create the bubble chart with discrete colors
fig_source_conv = px.scatter(
    combined_data,
    x='Lead Source Original',  # X-axis: Lead Source Original
    y='conversion_rate',  # Y-axis: Conversion rate
    size='total_mqls',  # Bubble size: total MQLs
    color='Year',  # Color by Year (discrete colors)
    #color_discrete_sequence=px.colors.qualitative.Set1,  # Use a qualitative color scheme for discrete colors
    hover_data={'total_mqls': True, 'converted_leads': True},  # Show additional data on hover
    text='conversion_rate',  # Display conversion rate as text
    title=f"YTD Conversion Rates (01 to {end_month:02d}-{end_year}) vs YTD {get_previous_year(end_year)}",
    labels={'conversion_rate': 'Conversion Rate (%)', 'Lead Source Original': 'Lead Source'},
    size_max=60  # Max bubble size for better visibility
)

# Customize layout
fig_source_conv.update_traces(
    texttemplate='%{text:.0f}%',  # Show percentage values
    textposition='middle center'  # Position the text inside the bubbles
)

# Add annotation explaining the bubble size
fig_source_conv.add_annotation(
    text="(Bubble size represents number of MQLs)",
    xref="paper", yref="paper",
    x=-.05, y=1.05, showarrow=False,
    font=dict(size=14),
    align="left"
)

# Update layout
fig_source_conv.update_layout(
    yaxis_title="Conversion Rate (%)",
    xaxis_title="Lead Source",
    showlegend=True,
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    margin=dict(t=60, b=40, l=40, r=40)
)

# Display the plot in the Streamlit app
st.plotly_chart(fig_source_conv, key='11a')
#plotly.offline.plot(fig_source_conv)

#%%What is the average lead scoring for each lead source?
st.subheader('12. What is the average lead scoring for each lead source?', divider='rainbow')

# Step 1: Group by 'Lead Source Original' and calculate the average lead score
avg_lead_score_data = filtered_data.groupby('Lead Source Original')['[Lead Score]'].mean().reset_index()

# Step 2: Rename columns for clarity
avg_lead_score_data.columns = ['Lead Source Original', 'Average Lead Score']

# Step 3: Create a bar chart to visualize average lead score for each lead source
fig_lead_score = px.bar(avg_lead_score_data, 
             x='Lead Source Original', 
             y='Average Lead Score', 
             text='Average Lead Score',
             title="Average Lead Score by Lead Source",
             labels={'Average Lead Score': 'Average Lead Score', 'Lead Source Original': 'Lead Source'})

# Customize the layout for better visualization
fig_lead_score.update_traces(texttemplate='%{text:.2f}', textposition='outside')

fig_lead_score.update_layout(
    yaxis_title="Average Lead Score",
    xaxis_title="Lead Source",
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    margin=dict(t=60, b=40, l=40, r=40)
)

st.plotly_chart(fig_lead_score, key='12a')

#%%What are the monthly trends in lead creation?
st.subheader('13. What are the monthly trends in lead creation?', divider='rainbow')

# Step 1: Filter the data where 'YTD MQLs' is equal to 1
filtered_data_mql = filtered_data[filtered_data['YTD MQLs'] == 1]

# Step 2: Extract the month and year from the 'Month-Year' column

# Extract the month and year separately for easier plotting
filtered_data_mql['Month'] = filtered_data_mql['Year-Month'].dt.strftime('%b')  # Extract month name (Jan, Feb, etc.)
filtered_data_mql['Year'] = filtered_data_mql['Year-Month'].dt.year  # Extract year

# Step 3: Group by 'Month' and 'Year' and compute the sum of 'YTD MQLs'
mqls_sum = filtered_data_mql.groupby(['Year', 'Month']).agg({'YTD MQLs': 'sum'}).reset_index()

# Since we need a consistent x-axis from Jan to Dec, we use the month numbers for sorting
months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Convert 'Month' to a categorical type with the correct order
mqls_sum['Month'] = pd.Categorical(mqls_sum['Month'], categories=months_order, ordered=True)

# Step 5: Sort the DataFrame by 'Month' to ensure correct order
mqls_sum = mqls_sum.sort_values('Month')

#Filter for 2022
mqls_sum = mqls_sum.loc[mqls_sum['Year']>2022,].reset_index(drop=True)

# Step 4: Plot the data using Plotly
fig_year_mql = px.line(
    mqls_sum,
    x='Month',
    y='YTD MQLs',
    color='Year',
    category_orders={'Month': months_order},  # Ensure the x-axis is ordered from Jan to Dec
    title='MQLs per Month for each Year',
    labels={'YTD MQLs': 'YTD MQLs', 'Month': 'Month'},
    markers=True
)

# Update the layout for better readability
fig_year_mql.update_layout(
    xaxis_title="Month",
    yaxis_title="Count of MQLs",
    width=800,
    height=600,
    legend_title="Year"
)

# Step 5: Display the plot in Streamlit
#plotly.offline.plot(fig_year_mql)
st.plotly_chart(fig_year_mql, key='13a')

#%%What are the monthly trends in lead creation?
st.subheader('14. What is the average order amount associated with converted leads?', divider='rainbow')

# Step 2: Count leads and conversions (opportunities) by zone
zones_amount_ts = filtered_data_mql.groupby(['Year-Month', 'Zone']).agg(
    total_order_amount=('Order Amount', lambda x: x.sum()),
).reset_index()
zones_amount_ts['Year-Month'] = zones_amount_ts['Year-Month'].astype(str)

fig_zone_amt = create_stacked_bar_ts(
    data=zones_amount_ts,
    time_column='Year-Month',
    value_column='total_order_amount',
    title='Order Amount Over Time by Zone',
    yaxis_title='Order Amount',
    color_column='Zone'  # Stacked bars by Zone
)

# Show the plot
st.plotly_chart(fig_zone_amt, key='14a')

# Count leads and conversions (opportunities) by BU
bu_amount_ts = filtered_data.groupby(['Year-Month', 'Business Unit']).agg(
    total_order_amount=('Order Amount', lambda x: x.sum()),
).reset_index()
bu_amount_ts['Year-Month'] = bu_amount_ts['Year-Month'].astype(str)

fig_bu_amt = create_stacked_bar_ts(
    data=bu_amount_ts,
    time_column='Year-Month',
    value_column='total_order_amount',
    title='Order Amount Over Time by Business Unit',
    yaxis_title='Order Amount',
    color_column='Business Unit'  # Stacked bars by Zone
)

# Show the plot
st.plotly_chart(fig_bu_amt, key='14b')
#plotly.offline.plot(fig_order)

#%%What are the monthly trends in lead creation?
st.subheader('15. How many leads are generated per campaign channel?', divider='rainbow')

# Step 1: Group by 'Campaign - Channel' and count the number of leads
channel_leads = filtered_data.groupby('[Campaign - Channel]').agg(
    total_mqls=('YTD MQLs', lambda x: x.sum()),
).reset_index()

# Step 2: Create a bar chart to visualize the number of leads per campaign channel
fig_cc = px.bar(
    channel_leads, 
    x='[Campaign - Channel]', 
    y='total_mqls', 
    title="MQLs Generated per Campaign Channel",
    labels={'total_mqls': 'Total MQLs', '[Campaign - Channel]': 'Campaign Channel'},
    text='total_mqls'  # Display the lead count on the bars
)

# Customize layout
fig_cc.update_traces(texttemplate='%{text}', textposition='outside')

fig_cc.update_layout(
    yaxis_title="Total MQLs",
    xaxis_title="Campaign Channel",
    xaxis={'categoryorder': 'total descending'},
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    margin=dict(t=60, b=100, l=40, r=40),
    height=500  # Adjust height if needed
)

# Show the plot
st.plotly_chart(fig_cc, key='15a')

#%%Which campaigns contribute the most to closed opportunities?
st.subheader('16. Which campaigns contribute the most to closed opportunities?', divider='rainbow')

# Step 1: Filter the dataframe for closed opportunities based on 'Opportunity Stage'
closed_opportunities = filtered_opp[~filtered_opp['Opportunity Stage'].isin(['ACTIVE'])]

# Step 2: Group by 'Campaign Name' and sum the closed opportunities for each campaign
campaign_closed_opportunities = closed_opportunities.groupby('Campaign Name').agg(
    total_opp_value=('Opportunity Amount Updated', 'sum')
).reset_index()

# Step 3: Sort the data by opportunity amount in descending order for better visualization
campaign_closed_opportunities = campaign_closed_opportunities.sort_values(by='total_opp_value', ascending=False)
campaign_closed_opportunities['total_opp_value_formatted'] = campaign_closed_opportunities['total_opp_value'].apply(format_euro)

# Step 4: Add a column for cumulative percentage for Pareto calculation
campaign_closed_opportunities['cumulative_opp_value'] = campaign_closed_opportunities['total_opp_value'].cumsum()
campaign_closed_opportunities['cumulative_percent'] = (campaign_closed_opportunities['cumulative_opp_value'] / 
                                                       campaign_closed_opportunities['total_opp_value'].sum()) * 100

# Step 5: Implement toggle for Pareto functionality (show campaigns contributing to 80% of opportunity amount)
pareto_camp_amt = st.toggle("Pareto View", key="q26")

if pareto_camp_amt:
    # Filter campaigns that contribute to 80% of opportunity amount
    campaign_closed_opportunities = campaign_closed_opportunities[campaign_closed_opportunities['cumulative_percent'] <= 80]

# Step 6: Create a horizontal bar chart
fig_camp_contrb = px.bar(
    campaign_closed_opportunities, 
    x='total_opp_value',  # Now on x-axis
    y='Campaign Name',  # Campaign names on y-axis
    orientation='h',  # Set to horizontal
    title="Campaigns Contributing to Closed Opportunities",
    labels={'total_opp_value': 'Closed Opportunity Amount (€)', 'Campaign Name': 'Campaign Name'},
    text='total_opp_value_formatted'  # Display the formatted closed opportunity amount on the bars
)

# Customize the layout
fig_camp_contrb.update_traces(texttemplate='%{text}', textposition='outside')

fig_camp_contrb.update_layout(
    xaxis_title="Opportunity Amount (€)",  # Change to x-axis since the orientation is horizontal
    yaxis_title="Campaign Name",
    yaxis={'categoryorder':'total ascending'},
    margin=dict(t=60, b=100, l=40, r=40),
    height=1000,
    width=1000
)

# Show the plot
st.plotly_chart(fig_camp_contrb, key='16a')


#%%What is the time taken from lead creation to order promised date?
st.subheader('17. What is the time taken from lead creation to order promised date?', divider='rainbow')

# Step 1: Ensure 'Lead Creation Date' and 'Opportunity Order Promised Date' are in datetime format
filtered_data['Lead Creation Date'] = pd.to_datetime(filtered_data['Lead Creation Date'])
filtered_data['Opportunity Order Promised Date'] = pd.to_datetime(filtered_data['Opportunity Order Promised Date'])

# Step 2: Calculate the time difference in days from 'Lead Creation Date' to 'Opportunity Order Promised Date'
filtered_data['Time Difference (Days)'] = (filtered_data['Opportunity Order Promised Date'] - filtered_data['Lead Creation Date']).dt.days

# Step 3: Group by 'Business Unit' and 'Zone' to calculate the mean time difference
time_diff_by_bu_zone = filtered_data.groupby(['Business Unit', 'Zone']).agg(
    mean_time_diff=('Time Difference (Days)', lambda x: round(x.mean(), 0))
).reset_index()

# Step 4: Pivot the data to create a matrix for the heatmap
heatmap_data = time_diff_by_bu_zone.pivot(index='Business Unit', columns='Zone', values='mean_time_diff')

# Step 5: Create the heatmap using px.imshow to include annotations
fig_crt_heat = px.imshow(
    heatmap_data,
    color_continuous_scale='Viridis',
    text_auto=True,  # Automatically adds rounded value labels
    aspect="auto",  # Adjust aspect ratio
    labels=dict(x="Zone", y="Business Unit", color="Mean Time (Days)")
)

# Step 6: Customize layout for better readability
fig_crt_heat.update_layout(
    title="Mean Time Difference from Lead Creation to Order Promised by Business Unit and Zone",
    xaxis_title="Zone",
    yaxis_title="Business Unit",
    coloraxis_colorbar=dict(title="Mean Time (Days)"),
    margin=dict(t=60, b=100, l=40, r=40),
    height=700  # Adjust height if needed
)

# Show the heatmap
st.plotly_chart(fig_crt_heat, key='17t')

#%%How often do leads in a specific business category convert?
st.subheader('18. How often do leads in a specific business category convert (or ACTIVE)?', divider='rainbow')

# Step 2: Add a column to flag whether the opportunity is closed
filtered_data['Converted'] = ~filtered_data['Opportunity Stage'].isin(['ACTIVE'])

# Step 3: Group by 'Business Unit' and calculate total leads and total closed opportunities
conversion_data = filtered_data.groupby('Business Unit').agg(
    total_leads=('Lead ID', 'size'),  # Count total leads for each Business Unit
    total_converted=('Converted', 'sum')  # Count converted leads for each Business Unit
).reset_index()

# Step 4: Calculate conversion rate as a percentage
conversion_data['Conversion Rate (%)'] = (conversion_data['total_converted'] / conversion_data['total_leads']) * 100

# Remove the ones with zero values
conversion_data = conversion_data[conversion_data['Conversion Rate (%)'] > 0]

# Step 5: Create a custom label for each bar
conversion_data['Label'] = conversion_data.apply(
    lambda row: f"{row['Conversion Rate (%)']:.0f}% (Leads: {row['total_leads']})", axis=1
)

# Step 6: Create a horizontal bar chart to visualize the conversion rate by Business Unit
fig_biz_conv = px.bar(
    conversion_data,
    y='Business Unit',  # Business Unit on the y-axis (for horizontal orientation)
    x='Conversion Rate (%)',  # Conversion rate on the x-axis
    text='Label',  # Display the custom label (Conversion Rate + Total Leads)
    orientation='h',  # Set orientation to horizontal
    labels={'Conversion Rate (%)': 'Conversion Rate (%)', 'Business Unit': 'Business Unit'}
)

# Customize layout to show labels and adjust appearance
fig_biz_conv.update_traces(
    textposition='inside'  # Display the labels outside the bars for better readability
)

# Customize layout
fig_biz_conv.update_layout(
    xaxis_title="Conversion Rate (%)",
    yaxis_title="Business Unit",
    yaxis={'categoryorder': 'total ascending'},  # Sort y-axis in descending order based on conversion rate
    margin=dict(t=60, b=100, l=40, r=40),
    height=600  # Adjust height as needed
)

# Show the plot
st.plotly_chart(fig_biz_conv, key='18a')

#%%Which user types (e.g., admin, sales) close the most opportunities?
st.subheader('19. Which user types (e.g., admin, sales) close the most opportunities?', divider='rainbow')
# Step 1: Filter the dataframe for closed opportunities based on 'Opportunity Stage'
closed_opportunities = filtered_opp[~filtered_opp['Opportunity Stage'].isin(['ACTIVE'])]

# Step 2: Group by 'Lead Role' and calculate the number of closed opportunities and total opportunity value for each role
role_closed_opportunities = closed_opportunities.groupby('Lead Role').agg(
    Closed_Opportunity_Count=('Golden Opportunity ID', 'size'),
    Total_Opportunity_Value=('Opportunity Amount Updated', 'sum')
).reset_index()

# Format the opp_val_per_opp for display as Euro value
role_closed_opportunities['Total_Opportunity_Value'] = role_closed_opportunities['Total_Opportunity_Value'].map("{:.2}".format)

# Step 3: Sort the data by closed opportunity count in descending order for better visualization
role_closed_opportunities = role_closed_opportunities.sort_values(by='Closed_Opportunity_Count', ascending=True)

# Step 4: Create a bar chart for Closed Opportunity Count using Plotly Express
fig_role = px.bar(
    role_closed_opportunities,
    y='Lead Role',
    x='Closed_Opportunity_Count',
    text='Closed_Opportunity_Count',
    #title="User Roles Contributing to Closed Opportunities",
    labels={'Closed_Opportunity_Count': 'Closed Opportunity Count', 'Lead Role': 'User Role'},
    orientation='h'  # Horizontal orientation
)

# Step 5: Add markers for Opportunity Value using secondary x-axis
fig_role.update_layout(
    xaxis=dict(
        title="Closed Opportunity Count",
        titlefont=dict(color="blue"),
        tickfont=dict(color="blue"),
    ),
    xaxis2=dict(
        title="Total Opportunity Value (€)",
        titlefont=dict(color="orange"),
        tickfont=dict(color="orange"),
        anchor='y',  # Make both axes share the same y-axis
        overlaying='x',  # Overlay x2 axis on top of x1
        side='top'  # Position x2 at the top
    ),
    yaxis_title="User Role",
    height=800,
    width=1000,
    margin=dict(t=60, b=100, l=40, r=40),
    legend=dict(x=0.05, y=1.1, orientation='h')  # Adjust legend position
)

# Step 6: Add Opportunity Value markers on x2 axis (secondary x-axis)
for i, row in role_closed_opportunities.iterrows():
    fig_role.add_scatter(
        x=[row['Total_Opportunity_Value']],  # Use opportunity value for x2
        y=[row['Lead Role']],  # Same y-axis (Lead Role)
        mode='markers',
        marker=dict(color='orange', size=10),
        xaxis='x2',  # Reference the secondary x-axis
        name="Total Opportunity Value (€)",
        showlegend=False
    )

# Step 7: Display the plot in Streamlit
st.plotly_chart(fig_role, key='19a')
