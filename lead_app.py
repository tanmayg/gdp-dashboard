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
from pathlib import Path


# Function to wrap labels
def wrap_labels(labels, max_length=25):
    wrapped_labels = []
    for label in labels:
        wrapped_label = '<br>'.join([label[i:i+max_length] for i in range(0, len(label), max_length)])
        wrapped_labels.append(wrapped_label)
    return wrapped_labels

# Load your dataframe
@st.cache_data
def load_data():
    with open(Path(__file__).parent/'data/lead_app_data.pkl', 'rb') as file:
        loaded_dataframes = pickle.load(file)
    return loaded_dataframes

lead_univ = load_data()['x']
lead_data_opp = load_data()['y']

# Replace '-' with np.nan in the 'Opportunity ID' column
lead_univ = lead_univ.replace('-', np.nan)

#clean leadscore column
lead_univ['[Lead Score]'] = lead_univ['[Lead Score]'].str.replace('%', '')
lead_univ['[Lead Score]'] = lead_univ['[Lead Score]'].astype('float')
#lead_univ['Opportunity ID'] = lead_univ['Opportunity ID'].replace('-', np.nan)

# Initialize session state for the filters
if 'start_date' not in st.session_state:
    st.session_state.start_date = lead_univ['Lead Creation Date'].min()
if 'end_date' not in st.session_state:
    st.session_state.end_date = lead_univ['Lead Creation Date'].max()

# Streamlit app code
st.title('Leads Data Analysis')

# Sidebar filters
st.sidebar.header('Filters')

# Date filter using session state values
start_date = st.sidebar.date_input('Start date', lead_univ['Lead Creation Date'].min())
end_date = st.sidebar.date_input('End date', lead_univ['Lead Creation Date'].max())

# Create sidebar multi-select filter for 'Business Unit'
zone_filter = st.sidebar.multiselect(
    'Select Zone', 
    options=lead_univ['Zone'].unique(), 
    default=lead_univ['Zone'].unique()
)

# Add Clear All button
if st.sidebar.button('Clear All'):
    st.session_state.start_date = lead_univ['Lead Creation Date'].min()
    st.session_state.end_date = lead_univ['Lead Creation Date'].max()
    st.rerun()  # Refresh the app to apply the reset values

# Filter data by date range
filtered_data = lead_univ[(lead_univ['Lead Creation Date'] >= start_date) & (lead_univ['Lead Creation Date'] <= end_date)]
filtered_data = filtered_data[filtered_data['Zone'].isin(zone_filter)]

filtered_opp = lead_data_opp[(lead_data_opp['Lead Creation Date'] >= start_date) & (lead_data_opp['Lead Creation Date'] <= end_date)]
filtered_opp = filtered_opp[filtered_opp['Zone'].isin(zone_filter)]

# Group by Zone and count leads
zone_lead_counts = filtered_data['Zone'].value_counts().reset_index()
zone_lead_counts.columns = ['Zone', 'Lead Count']

# Sort by lead count in descending order
zone_lead_counts = zone_lead_counts.sort_values(by='Lead Count', ascending=False)

#%% Subheader for Q1
st.subheader('1. Which zones generate the most leads?', divider='rainbow')
# Create the bar chart using plotly
fig = px.bar(zone_lead_counts, 
             x='Zone', 
             y='Lead Count', 
             text='Lead Count', 
             title='Number of Leads by Zone', 
             labels={'Zone': 'Zone', 'Lead Count': 'Number of Leads'})

# Adjust layout to show bars in descending order
fig.update_traces(textposition='outside')
fig.update_layout(xaxis_title='Zone', 
                  yaxis_title='Number of Leads', 
                  xaxis={'categoryorder': 'total descending'})

# Display the plot in the Streamlit app
st.plotly_chart(fig)

#%% Subheader for Q2
st.subheader('2. What are the most common lead sources?', divider='rainbow')

# Create pie charts for 'Lead Source' and 'Lead Source Original'
lead_source_counts = filtered_data['Lead Source'].value_counts().reset_index()
lead_source_counts.columns = ['Lead Source', 'Count']

# Create sunburst plot for original category
lead_catg_counts = filtered_data['Lead_Source_Catg'].value_counts().reset_index()
lead_catg_counts.columns = ['Lead_Category', 'Count']

# Add a 'Parent' column for the hierarchy
lead_catg_counts['Parent'] = lead_catg_counts['Lead_Category'].apply(lambda x: 'Lead Source' if x == 'Event' else 'Digital')

# Now, manually append the parent 'Digital' and 'Lead Source' rows to complete the hierarchy

lead_catg_counts = pd.concat([lead_catg_counts, pd.DataFrame({'Lead_Category': 'Digital', 'Count': lead_catg_counts[lead_catg_counts['Parent'] == 'Digital']['Count'].sum(), 'Parent': 'Lead Source'}, index=[0])]).reset_index(drop=True)

lead_catg_counts = pd.concat([lead_catg_counts, pd.DataFrame({'Lead_Category': 'Lead Source', 'Count': lead_catg_counts[lead_catg_counts['Parent'] == 'Lead Source']['Count'].sum(), 'Parent': ''}, index=[0])]).reset_index(drop=True)

# Prepare the Sunburst Chart
fig = go.Figure(go.Sunburst(
    labels=lead_catg_counts['Lead_Category'],
    parents=lead_catg_counts['Parent'],
    values=lead_catg_counts['Count'],
    branchvalues="total",
    hoverinfo="label+percent parent+value"
))

# Update layout for better display
fig.update_layout(
    margin=dict(t=0, l=0, r=0, b=0),
    title_text="Lead Source Hierarchy"
)

# Display the pie charts in the Streamlit app
st.plotly_chart(fig)

#%% Subheader for Q3
st.subheader('3. What proportion of leads are getting converted to opportunities by zone?', divider='rainbow')

# Count leads and conversions (opportunities) by zone
conversion_data = filtered_data.groupby('Zone').agg(
    total_leads=('Lead ID', 'count'),
    converted_leads=('Opportunity ID', lambda x: x.notna().sum())
).reset_index()

# Calculate proportion of converted leads
conversion_data['conversion_rate'] = conversion_data['converted_leads'] / conversion_data['total_leads']

# Sort the data by total leads in descending order
conversion_data = conversion_data.sort_values(by='total_leads', ascending=False)

# Melt data to get it in long format for stacked bar plot
melted_data = pd.melt(conversion_data, id_vars='Zone', value_vars=['converted_leads', 'total_leads'],
                      var_name='Lead Status', value_name='Count')

# Adjust for stacked bar chart where 'converted_leads' needs to be shown as converted portion of the total
melted_data['Lead Status'] = melted_data['Lead Status'].map({
    'converted_leads': 'Opportunities',
    'total_leads': 'Leads'
})

# Create the stacked bar chart using plotly
fig = px.bar(melted_data, 
             x='Zone', 
             y='Count', 
             color='Lead Status', 
             text='Count', 
             title='Proportion of Leads Converted to Opportunities by Zone', 
             labels={'Zone': 'Zone', 'Count': 'Number of Leads'},
             barmode='overlay')  # Stack the bars to show proportion

# Update layout
fig.update_layout(xaxis_title='Zone', 
                  yaxis_title='Number of Leads', 
                  legend_title='Lead Status',
                  height=600)

# Display the plot in the Streamlit app
st.plotly_chart(fig)

# Show conversion rates for each zone as a table
#st.subheader('Conversion Rates by Zone')
st.markdown("<h3 style='font-size:18px;'>Conversion Rates by Zone</h3>", unsafe_allow_html=True)
# Show conversion rates as a percentage
conversion_data['conversion_rate'] = (conversion_data['conversion_rate']*100).round(2)
st.dataframe(conversion_data[['Zone', 'conversion_rate']].rename(columns={'conversion_rate': 'Conversion Rate (%)'}), hide_index=True)

#%% Subheader for Q4
st.subheader('4. What is the average time to convert a Marketing Qualified Lead (MQL) to an opportunity?', divider='rainbow')

# Replace '-' with np.nan in the 'Opportunity ID' column
filtered_data['MQL to Opp. Lead Velocity (Days)'] = filtered_data['MQL to Opp. Lead Velocity (Days)'].replace('-', np.nan)
filtered_data['MQL to Opp. Lead Velocity (Days)'] = filtered_data['MQL to Opp. Lead Velocity (Days)'].astype(float)
# Compute average time to convert MQL to opportunity (MQL to Opp. Lead Velocity (Days)) for each zone
conversion_data['avg_conversion_time'] = filtered_data.groupby('Zone')['MQL to Opp. Lead Velocity (Days)'].mean().reset_index(drop=True)
conversion_data['avg_conversion_time'] = conversion_data['avg_conversion_time'].round(2)

# Sort the data by total leads (this is already done above, so zones will be in correct order)

# Create a bar chart to display average conversion time by zone
fig_avg_conversion = px.bar(
    conversion_data, 
    x='Zone', 
    y='avg_conversion_time', 
    text='avg_conversion_time', 
    title='Average Time to Convert MQL to Opportunity by Zone',
    labels={'Zone': 'Zone', 'avg_conversion_time': 'Avg Conversion Time (Days)'}
)

# Update layout to improve display
fig_avg_conversion.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_avg_conversion.update_layout(
    xaxis_title='Zone', 
    yaxis_title='Average Conversion Time (Days)', 
    coloraxis_showscale=False,  # Hide the color scale
    height=600
)

# Display the plot in the Streamlit app
st.plotly_chart(fig_avg_conversion)

# Display updated conversion_data dataframe (including the new average time column)
st.markdown("<h3 style='font-size:18px;'>Conversion Data by Zone</h3>", unsafe_allow_html=True)
st.dataframe(conversion_data[['Zone', 'total_leads', 'converted_leads', 'conversion_rate', 'avg_conversion_time']].rename(columns={
    'conversion_rate': 'Conversion Rate (%)',
    'avg_conversion_time': 'Avg Conversion Time (Days)'
}), hide_index=True)

#%% Subheader for Q5
st.subheader('5. Which campaigns result in the highest lead conversion rates?', divider='rainbow')

# Group by 'Campaign Name' and calculate total leads and converted leads
campaign_conversion_data = filtered_data.groupby('Campaign Name').agg(
    total_leads=('Lead ID', 'count'),
    converted_leads=('Opportunity ID', lambda x: x.notna().sum())
).reset_index()

# Calculate conversion rate
campaign_conversion_data['conversion_rate'] = (campaign_conversion_data['converted_leads'] / campaign_conversion_data['total_leads']) * 100
# Filter rows where conversion rate > 0
campaign_conversion_data = campaign_conversion_data[campaign_conversion_data['conversion_rate'] > 0]

# Sort the data by total_leads in descending order
campaign_conversion_data = campaign_conversion_data.sort_values(by='conversion_rate', ascending=False)

# Create a bar chart to display the conversion rate by campaign
fig_campaign_conversion = px.bar(
    campaign_conversion_data, 
    x='Campaign Name', 
    y='conversion_rate', 
    text='conversion_rate', 
    title='Lead Conversion Rates by Campaign',
    labels={'Campaign Name': 'Campaign Name', 'conversion_rate': 'Conversion Rate (%)'},
    color='conversion_rate',
    color_continuous_scale=px.colors.sequential.Viridis
)

# Update layout for better display
fig_campaign_conversion.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_campaign_conversion.update_layout(
    xaxis_title='Campaign Name', 
    yaxis_title='Conversion Rate (%)', 
    xaxis={'categoryorder':'total descending'},  # Sort the x-axis based on total_leads
    height=900,
    showlegend=False
)

# Display the plot in the Streamlit app
st.plotly_chart(fig_campaign_conversion)

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
on = st.toggle("Pareto View")

if on:
    fig_pareto_ov = px.bar(
    pareto_data, 
    x='Campaign Name', 
    y='total_order_value', 
    text='total_order_value', 
    title='Total Order Value by Campaign (Pareto)',
    labels={'Campaign Name': 'Campaign Name', 'total_order_value': 'Total Order Value (€)'},
    )

    # Update layout for better display
    fig_pareto_ov.update_traces(texttemplate='%{text:.3s}', textposition='inside')
    fig_pareto_ov.update_layout(
        xaxis_title='Campaign Name', 
        yaxis_title='Total Order Value (€)', 
        xaxis={'categoryorder':'total descending'},  # Sort the x-axis based on total order value
        height=900,
        showlegend=False
    )
    st.plotly_chart(fig_pareto_ov)
else:
    fig_campaign_ov = px.bar(
    campaign_ov_data, 
    x='Campaign Name', 
    y='total_order_value', 
    text='total_order_value', 
    title='Total Order Value by Campaign',
    labels={'Campaign Name': 'Campaign Name', 'total_order_value': 'Total Order Value (€)'},
    color='total_order_value',
    color_continuous_scale=px.colors.sequential.Blues
    )

    # Update layout for better display
    fig_campaign_ov.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_campaign_ov.update_layout(
        xaxis_title='Campaign Name', 
        yaxis_title='Total Order Value (€)', 
        xaxis={'categoryorder':'total descending'},  # Sort the x-axis based on total order value
        height=900,
        showlegend=False
    )

    st.plotly_chart(fig_campaign_ov)

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

# Display the plot in the Streamlit app
st.plotly_chart(fig_bu_lead_quality)

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

# 3. Display the result of the Chi-Square test
st.markdown("<h3 style='font-size:18px;'>Chi-Square Test Results: Business Unit vs Lead Qualification Level</h3>", unsafe_allow_html=True)
st.write(f"Chi-Square Statistic: {chi2_stat_bu:.2f}")
st.write(f"Degrees of Freedom: {dof_bu}")
st.write(f"P-Value: {p_val_bu:.4f}")

# Conclusion based on the P-value
if p_val_bu < 0.05:
    st.write("There is a significant relationship between Business Unit and Lead Qualification Level (Reject H0).")
else:
    st.write("There is no significant relationship between Business Unit and Lead Qualification Level (Fail to Reject H0).")

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
st.plotly_chart(fig_residuals_bu)

# 6. Highlight significant residuals (residuals > 1.96 or < -1.96 are often considered significant at p < 0.05)
st.markdown("<h3 style='font-size:18px;'>Significant Residuals (Indicating Specific Relationships)</h3>", unsafe_allow_html=True)
significant_residuals_bu = residuals_bu_flat[(residuals_bu_flat['Residual'] > 1.96) | (residuals_bu_flat['Residual'] < -1.96)]
st.dataframe(significant_residuals_bu)

st.write('Positive residuals indicate a higher observed count than expected, while negative residuals indicate a lower observed count than expected.')


#%%# Subheader for Q8
st.subheader('8. What are the Main Reasons for Lead Rejection?')

# Group by 'Reason for Rejection' and count occurrences
reason_rejection_data = filtered_data['Reason for Rejection'].value_counts().reset_index()
reason_rejection_data.columns = ['Reason for Rejection', 'Lead Count']

# Plot bar chart for reasons of rejection
fig_rejection_reasons = px.bar(
    reason_rejection_data,
    x='Reason for Rejection',
    y='Lead Count',
    text='Lead Count',
    title='Reasons for Lead Rejection',
    labels={'Reason for Rejection': 'Rejection Reason', 'Lead Count': 'Number of Leads'}
)

# Update layout for better display
fig_rejection_reasons.update_traces(textposition='outside')
fig_rejection_reasons.update_layout(
    xaxis_title='Reason for Rejection',
    yaxis_title='Number of Leads',
    xaxis_tickangle=-45,
    height=600
)

# Display the plot
st.plotly_chart(fig_rejection_reasons)

#%% Subheader for Q9
st.subheader('9. How effective are different lead actions in converting leads to opportunities?')
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
st.plotly_chart(fig_lead_action_stage)

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
st.markdown("<h3 style='font-size:18px;'>Chi-Square Test Results</h3>", unsafe_allow_html=True)
st.write(f"Chi-Square Statistic: {chi2_stat:.2f}")
st.write(f"Degrees of Freedom: {dof}")
st.write(f"P-Value: {p_val:.4f}")

# Conclusion based on the P-value
if p_val < 0.05:
    st.write("There is a significant relationship between Lead Action and Opportunity Stage (Reject H0).")
else:
    st.write("There is no significant relationship between Lead Action and Opportunity Stage (Fail to Reject H0).")

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
st.plotly_chart(fig_residuals)

# 6. Highlight significant residuals (residuals > 1.96 or < -1.96 are often considered significant at p < 0.05)
st.markdown("<h3 style='font-size:18px;'>Significant Residuals (Indicating Specific Relationships)</h3>", unsafe_allow_html=True)
significant_residuals = residuals_flat[(residuals_flat['Residual'] > 1.96) | (residuals_flat['Residual'] < -1.96)]
st.dataframe(significant_residuals)

st.write('Positive residuals indicate a higher observed count than expected, while negative residuals indicate a lower observed count than expected.')

#%% Subheader for Q10
st.subheader('10. What are the popular products of interest among leads?')

# Count occurrences of each product
product_counts = filtered_data['Product of Interest'].value_counts().reset_index()
product_counts.columns = ['Product of Interest', 'Lead Count']

# Sort by Lead Count in descending order
product_counts = product_counts.sort_values(by='Lead Count', ascending=False)

# Plot the most popular products using a bar chart
st.subheader('Popular Products of Interest Among Leads', divider='rainbow')
fig_product_interest = px.bar(product_counts, 
                              x='Product of Interest', 
                              y='Lead Count', 
                              text='Lead Count', 
                              title='Trending Products Among Leads',
                              labels={'Product of Interest': 'Product', 'Lead Count': 'Number of Leads'})

# Adjust layout to show bars in descending order
fig_product_interest.update_traces(textposition='outside')
fig_product_interest.update_layout(xaxis_title='Product of Interest', 
                                   yaxis_title='Number of Leads', 
                                   xaxis={'categoryorder': 'total descending'},  # Sort by product popularity
                                   height=600)

# Display the plot in the Streamlit app
st.plotly_chart(fig_product_interest)

# Optionally, display the dataframe if needed
st.write("Product Interest Data", product_counts)

#%%12. Which lead sources have the highest conversion rates?
st.subheader('12. Which lead sources have the highest conversion rates?')

# Step 1: Group by 'Lead Source Original' and calculate conversion rates
conversion_data = filtered_data.groupby('Lead Source Original').agg(
    total_leads=('Lead Source Original', 'size'),
    converted_leads=('Opportunity ID', lambda x: x.notna().sum())
).reset_index()

# Step 2: Calculate conversion rate as a percentage
conversion_data['conversion_rate'] = round((conversion_data['converted_leads'] / conversion_data['total_leads']) * 100, 0)

# Step 4: Create a bubble chart with lead count as the bubble size
fig_source_conv = px.scatter(conversion_data, 
                 x='Lead Source Original', 
                 y='conversion_rate', 
                 size='total_leads',  # Bubble size represents total leads
                 text='conversion_rate',  # Show conversion rate as text
                 hover_data={'total_leads': True, 'converted_leads': True},  # Show extra info on hover
                 title="Conversion Rates and Total Leads by Lead Source",
                 labels={'conversion_rate': 'Conversion Rate (%)', 'Lead Source Original': 'Lead Source'},
                 size_max=60)  # Set max bubble size for visibility

# Customize the layout
fig_source_conv.update_traces(texttemplate='%{text:.0f}%', textposition='top center')

# Add an annotation to explain bubble size
fig_source_conv.add_annotation(
    text="Bubble size represents number of leads",
    xref="paper", yref="paper",
    x=1.05, y=1.1, showarrow=False,
    font=dict(size=14, color="black"),
    align="left"
)

fig_source_conv.update_layout(
    yaxis_title="Conversion Rate (%)",
    xaxis_title="Lead Source",
    showlegend=False,
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    margin=dict(t=60, b=40, l=40, r=40)
)

st.plotly_chart(fig_source_conv)

#%%What is the average lead scoring for each lead source?
st.subheader('13. What is the average lead scoring for each lead source?')

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

st.plotly_chart(fig_lead_score)

#%%Which users are generating the most leads?
st.subheader('14. Which users are generating the most leads?')

# Step 1: Group by 'Username' and count the number of leads for each user
user_lead_data = filtered_data.groupby('Username').size().reset_index(name='Lead Count')

# Step 2: Sort the data in descending order of lead count
user_lead_data = user_lead_data.sort_values(by='Lead Count', ascending=False)

# Step 3: Create a bar chart to visualize the total number of leads generated by each user
fig_user = px.bar(user_lead_data, 
             x='Username', 
             y='Lead Count', 
             text='Lead Count',
             title="Leads Generated by Each User",
             labels={'Lead Count': 'Total Leads', 'Username': 'User'})

# Customize the layout for better visualization
fig_user.update_traces(texttemplate='%{text}', textposition='outside')

fig_user.update_layout(
    yaxis_title="Total Leads",
    xaxis_title="User",
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    xaxis={'categoryorder': 'total descending'},
    margin=dict(t=60, b=40, l=40, r=40)
)

st.plotly_chart(fig_user)

#%%What is the distribution of leads by business category?
st.subheader('15. What is the distribution of leads by business category?')

# Step 1: Group by 'Business' and 'Business Category' and count the number of leads in each category
business_category_data = filtered_data.groupby(['Business', 'Business Category']).size().reset_index(name='Lead Count')
# Sort by lead count in descending order
business_category_data = business_category_data.sort_values(by=['Business', 'Lead Count'], ascending=[False, False])

# Step 2: Create a bar chart with colors based on 'Business' and only display 'Business Category' on the x-axis
fig_cat = px.bar(business_category_data, 
             x='Business Category', 
             y='Lead Count', 
             color='Business',  # Color bars based on 'Business'
             #title="Lead Distribution by Business and Business Category",
             labels={'Lead Count': 'Total Leads'},
             text='Lead Count')  # Show lead count on bars

# Step 3: Adjust the layout for better readability
fig_cat.update_traces(texttemplate='%{text}', textposition='outside')

# Remove the x-axis title
fig_cat.update_layout(
    yaxis_title="Total Leads",
    xaxis_title=None,  # Hide the 'Business Category' title
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    margin=dict(t=60, b=100, l=40, r=40),
    xaxis_tickangle=-45,  # Rotate labels if needed
    height=600,  # Adjust height based on the number of categories
    legend=dict(
        orientation="h",  # Make the legend horizontal
        yanchor="bottom", 
        y=1.0,  # Position the legend above the plot
        xanchor="center", 
        x=0.5,
        bgcolor='rgba(255, 255, 255, 0)',  # Set legend background to transparent
        bordercolor='rgba(255, 255, 255, 0)',  # Optional: set border to transparent
        font=dict(color='black')  # Optional: set font color for visibility
    )
)

st.plotly_chart(fig_cat)

#%%What are the monthly trends in lead creation?
st.subheader('18. What are the monthly trends in lead creation?')
# Step 1: Ensure 'Lead Creation Date' is in datetime format
filtered_data['Lead Creation Date'] = pd.to_datetime(filtered_data['Lead Creation Date'])

# Step 2: Extract 'Year' and 'Month' from the 'Lead Creation Date'
filtered_data['Year-Month'] = filtered_data['Lead Creation Date'].dt.to_period('M')  # Creates 'YYYY-MM' format

# Step 3: Group by 'Year-Month' and count the number of leads created each month
monthly_trends = filtered_data.groupby('Year-Month').size().reset_index(name='Lead Count')

# Step 4: Convert 'Year-Month' back to string for plotting
monthly_trends['Year-Month'] = monthly_trends['Year-Month'].astype(str)

# Step 5: Create a line chart to visualize the monthly trend in lead creation
fig_trend = px.line(monthly_trends, 
              x='Year-Month', 
              y='Lead Count', 
              title="Monthly Trends in Lead Creation",
              labels={'Year-Month': 'Month-Year', 'Lead Count': 'Total Leads'},
              markers=True)  # Add markers to the line plot

# Customize layout for readability
fig_trend.update_layout(
    xaxis_title="Month-Year",
    yaxis_title="Total Leads",
    xaxis_tickangle=-45,  # Rotate labels for readability
    margin=dict(t=60, b=100, l=40, r=40),
    height=600,  # Adjust height if needed
)

# Show the plot
st.plotly_chart(fig_trend)

#%%What are the monthly trends in lead creation?
st.subheader('19. What is the average order amount associated with converted leads?')

# Step 2: Group by 'Business' and calculate total leads and total order amount
business_summary = filtered_data.groupby('Business').agg(
    total_leads=('Lead ID', 'size'),  # Count total leads by Business
    total_order_amount=('Order Amount', 'sum')  # Sum of Order Amount by Business
).reset_index()

# Step 3: Create the figure object for plotting
fig_order = go.Figure()

# Step 4: Add bar trace for total leads (on primary y-axis)
fig_order.add_trace(
    go.Bar(
        x=business_summary['Business'],
        y=business_summary['total_leads'],
        name="Total Leads",
        marker_color='lightblue',
        yaxis='y1'  # Assign to primary y-axis
    )
)

# Step 5: Add line trace for total order amount (on secondary y-axis)
fig_order.add_trace(
    go.Scatter(
        x=business_summary['Business'],
        y=business_summary['total_order_amount'],
        name="Total Order Amount",
        mode='lines+markers+text',
        line=dict(color='orange', width=2),
        yaxis='y2',  # Assign to secondary y-axis
        text=business_summary['total_order_amount'],  # Show values on the line
        textposition='top right',
        texttemplate='%{text:.1s}',  # Format with one decimal and suffix like K/M
    )
)

# Step 6: Customize the layout for dual y-axes and apply cosmetic changes
fig_order.update_layout(
    title="Total Leads and Total Order Amount by Business",
    xaxis_title="Business",
    yaxis=dict(
        title="Total Leads",
        titlefont=dict(color="lightblue"),
        tickfont=dict(color="lightblue"),
        showgrid=False  # Hide grid lines on primary y-axis
    ),
    yaxis2=dict(
        title="Total Order Amount",
        titlefont=dict(color="orange"),
        tickfont=dict(color="orange"),
        overlaying='y',  # Overlay on same x-axis
        side='right',  # Display this axis on the right side
        showgrid=False  # Hide grid lines on secondary y-axis
    ),
    xaxis=dict(showgrid=False),  # Hide grid lines on x-axis
    legend=dict(x=0.5, xanchor='center', y=1.1, orientation='h'),
    margin=dict(t=60, b=100, l=40, r=40),
    height=500  # Adjust height if needed
)
# Show the plot
st.plotly_chart(fig_order)
#plotly.offline.plot(fig_order)

#%%What are the monthly trends in lead creation?
st.subheader('20. How many leads are generated per campaign channel?')

# Step 1: Group by 'Campaign - Channel' and count the number of leads
channel_leads = filtered_data.groupby('[Campaign - Channel]').agg(
    total_leads=('Lead ID', 'size')  # Count the total number of leads per channel
).reset_index()

# Step 2: Create a bar chart to visualize the number of leads per campaign channel
fig_cc = px.bar(
    channel_leads, 
    x='[Campaign - Channel]', 
    y='total_leads', 
    title="Leads Generated per Campaign Channel",
    labels={'total_leads': 'Total Leads', '[Campaign - Channel]': 'Campaign Channel'},
    text='total_leads'  # Display the lead count on the bars
)

# Customize layout
fig_cc.update_traces(texttemplate='%{text}', textposition='outside')

fig_cc.update_layout(
    yaxis_title="Total Leads",
    xaxis_title="Campaign Channel",
    xaxis={'categoryorder': 'total descending'},
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    margin=dict(t=60, b=100, l=40, r=40),
    height=500  # Adjust height if needed
)

# Show the plot
st.plotly_chart(fig_cc)

#%%Which clusters report the highest lead generation?
st.subheader('21. Which clusters report the highest lead generation?')

# Step 1: Group by 'Campaign - Channel' and count the number of leads
cluster_leads = filtered_data.groupby('Reporting Cluster').agg(
    total_leads=('Lead ID', 'size')  # Count the total number of leads per channel
).reset_index()

# Step 2: Create a bar chart to visualize the number of leads per campaign channel
fig_cl = px.bar(
    cluster_leads, 
    x='Reporting Cluster', 
    y='total_leads', 
    title="Leads Generated per Reporting Cluster",
    labels={'total_leads': 'Total Leads', 'Reporting Cluster': 'Cluster'},
    text='total_leads'  # Display the lead count on the bars
)

# Customize layout
fig_cl.update_traces(texttemplate='%{text}', textposition='outside')

fig_cl.update_layout(
    yaxis_title="Total Leads",
    xaxis_title="Cluster",
    xaxis={'categoryorder': 'total descending'},
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    margin=dict(t=60, b=100, l=40, r=40),
    height=500  # Adjust height if needed
)

# Show the plot
st.plotly_chart(fig_cl)

#%%How does lead quality vary by zone?
st.subheader('22. How does lead quality vary by zone?')

# Step 1: Group by 'Zone' and 'Lead Qualification Level' and count the number of leads for each combination
zone_qualification = filtered_data.groupby(['Zone', 'Lead Qualification Level']).size().reset_index(name='Lead Count')

# Step 2: Calculate the percentage of each qualification level within each zone
zone_totals = zone_qualification.groupby('Zone')['Lead Count'].transform('sum')  # Total leads per zone
zone_qualification['Percentage'] = (zone_qualification['Lead Count'] / zone_totals) * 100  # Percentage calculation

# Step 3: Create the stacked bar chart using plotly
fig_zone = px.bar(
    zone_qualification,
    x='Zone',
    y='Percentage',
    color='Lead Qualification Level',  # Stacking by qualification level
    title="Lead Qualification Distribution by Zone",
    labels={'Percentage': 'Qualification Level (%)', 'Zone': 'Zone'},
    text='Percentage',  # Show the percentage on the bars
)

# Customize the layout
fig_zone.update_traces(texttemplate='%{text:.1f}%', textposition='inside')

fig_zone.update_layout(
    yaxis_title="Percentage (%)",
    xaxis_title="Zone",
    barmode='stack',  # Stacked bars
    margin=dict(t=60, b=100, l=40, r=40),
    height=500,  # Adjust height as per your needs
    legend_title="Lead Qualification Level",
)

# Show the plot
st.plotly_chart(fig_zone)

#%%What is the lead-to-opportunity conversion rate for each business unit?
st.subheader('23. What is the lead-to-opportunity conversion rate for each business unit?')

# Step 1: Filter the dataframe to exclude leads without a valid 'Opportunity ID'
filtered_data['Opportunity Exists'] = filtered_data['Opportunity ID'].notna()  # Create a column to flag valid Opportunity IDs

# Step 2: Group by 'Business Unit' and calculate the total leads and total opportunities for each business unit
conversion_data = filtered_data.groupby('Business Unit').agg(
    total_leads=('Lead ID', 'size'),  # Count of total leads
    total_opportunities=('Opportunity Exists', 'sum')  # Count of valid opportunities
).reset_index()

# Step 3: Calculate the lead-to-opportunity conversion rate as a percentage
conversion_data['Conversion Rate (%)'] = (conversion_data['total_opportunities'] / conversion_data['total_leads']) * 100
#Sort the data by total_leads in descending order
conversion_data = conversion_data.sort_values(by='total_leads', ascending=False)

# Step 4: Create a bar chart to visualize the lead-to-opportunity conversion rate by business unit
fig_ratio = px.bar(
    conversion_data,
    x='Business Unit',
    y='Conversion Rate (%)',
    title="Lead-to-Opportunity Conversion Rate by Business Unit",
    labels={'Conversion Rate (%)': 'Conversion Rate (%)', 'Business Unit': 'Business Unit'},
    text='Conversion Rate (%)',  # Display the conversion rate on the bars
    hover_data={'total_leads': True}  # Show the total lead count on hover
)

# Step 5: Add the lead count to the bar text for better interpretation
fig_ratio.update_traces(
    texttemplate='Leads: %{customdata[0]}',  # Display both conversion rate and lead count
    customdata=conversion_data[['total_leads']],  # Pass lead count as custom data
    textposition='inside',  # Move the text inside the bar
    insidetextanchor='middle',  # Center the text inside the bar
)

# Customize the layout
fig_ratio.update_layout(
    yaxis_title="Conversion Rate (%)",
    xaxis_title="Business Unit",
    xaxis={'categoryorder': 'total descending'},
    margin=dict(t=60, b=100, l=40, r=40),
    height=700  # Adjust height if needed
)

# Show the plot
st.plotly_chart(fig_ratio)

#%%How do opportunity stages correlate with lead sources?
st.subheader('25. How do opportunity stages correlate with lead sources?')

# 1. Group by 'Lead Action' and 'Opportunity Stage' to get counts
lead_source_vs_stage = filtered_data.groupby(['Lead Source', 'Opportunity Stage']).size().reset_index(name='Count')

# 2. Plotting the stacked bar chart
fig_lead_source_stage = px.bar(
    lead_source_vs_stage,
    x='Opportunity Stage',
    y='Count',
    color='Lead Source',
    title='Effectiveness of Different Lead Sources in Converting to Opportunity Stages',
    labels={'Opportunity Stage': 'Opportunity Stage', 'Count': 'Lead Source', 'Lead Source': 'Lead Source'},
    barmode='stack'
)

# Adjust layout
fig_lead_source_stage.update_layout(
    xaxis_title='Opportunity Stage',
    yaxis_title='Number of Leads',
    xaxis={'categoryorder': 'total descending'},
    height=700
)

# Display the plot
st.plotly_chart(fig_lead_source_stage)

# 3. Perform Chi-Square test to assess the relationship between Lead Action and Opportunity Stage
# Create contingency table (cross-tabulation)
contingency_table = pd.crosstab(filtered_data['Lead Source'], filtered_data['Opportunity Stage'])

# Perform Chi-Square Test
chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)

# 4. Compute standardized residuals (Observed - Expected / sqrt(Expected))
observed = contingency_table.values
expected = ex
residuals = (observed - expected) / np.sqrt(expected)

# Display the result of the Chi-Square test
st.markdown("<h3 style='font-size:18px;'>Chi-Square Test Results</h3>", unsafe_allow_html=True)
st.write(f"Chi-Square Statistic: {chi2_stat:.2f}")
st.write(f"Degrees of Freedom: {dof}")
st.write(f"P-Value: {p_val:.4f}")

# Conclusion based on the P-value
if p_val < 0.05:
    st.write("There is a significant relationship between Lead Source and Opportunity Stage (Reject H0).")
else:
    st.write("There is no significant relationship between Lead Source and Opportunity Stage (Fail to Reject H0).")

# 5. Display residuals as a heatmap to identify specific relationships
residuals_df = pd.DataFrame(residuals, index=contingency_table.index, columns=contingency_table.columns)

# Flatten the residuals dataframe for plotting
residuals_flat = residuals_df.reset_index().melt(id_vars='Lead Source', var_name='Opportunity Stage', value_name='Residual')

# Plot residuals as heatmap using Plotly
fig_residuals = px.imshow(residuals_df, 
                          title="Standardized Residuals: Lead Source vs. Opportunity Stage",
                          labels=dict(color="Standardized Residual"),
                          aspect="auto")

# Display heatmap
st.plotly_chart(fig_residuals)

# 6. Highlight significant residuals (residuals > 1.96 or < -1.96 are often considered significant at p < 0.05)
st.markdown("<h3 style='font-size:18px;'>Significant Residuals (Indicating Specific Relationships)</h3>", unsafe_allow_html=True)
significant_residuals = residuals_flat[(residuals_flat['Residual'] > 1.96) | (residuals_flat['Residual'] < -1.96)]
st.dataframe(significant_residuals)

st.write('Positive residuals indicate a higher observed count than expected, while negative residuals indicate a lower observed count than expected.')

#%%Which campaigns contribute the most to closed opportunities?
st.subheader('26. Which campaigns contribute the most to closed opportunities?')

# Step 1: Filter the dataframe for closed opportunities based on 'StageName'
closed_stages = ['Order Booked', 'Sales Recognized', 'Order Promised']
closed_opportunities = filtered_data[filtered_data['StageName'].isin(closed_stages)]

# Step 2: Group by 'Campaign Name' and count the number of closed opportunities for each campaign
campaign_closed_opportunities = closed_opportunities.groupby('Campaign Name').size().reset_index(name='Closed Opportunity Count')

# Step 3: Sort the data by closed opportunity count in descending order for better visualization
campaign_closed_opportunities = campaign_closed_opportunities.sort_values(by='Closed Opportunity Count', ascending=False)

# Step 4: Create a bar chart to visualize the campaigns contributing the most to closed opportunities
fig_camp_contrb = px.bar(
    campaign_closed_opportunities, 
    x='Campaign Name', 
    y='Closed Opportunity Count', 
    title="Campaigns Contributing to Closed Opportunities",
    labels={'Closed Opportunity Count': 'Closed Opportunity Count', 'Campaign Name': 'Campaign Name'},
    text='Closed Opportunity Count'  # Display the closed opportunity count on the bars
)

# Customize the layout
fig_camp_contrb.update_traces(texttemplate='%{text}', textposition='outside')

fig_camp_contrb.update_layout(
    yaxis_title="Closed Opportunity Count",
    xaxis_title="Campaign Name",
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    margin=dict(t=60, b=100, l=40, r=40),
    height=500  # Adjust height as needed
)

# Show the plot
st.plotly_chart(fig_camp_contrb)

#%%What is the time taken from lead creation to order promised date?
st.subheader('27. What is the time taken from lead creation to order promised date?')

# Step 1: Ensure 'Lead Creation Date' and 'Opportunity Order Promised Date' are in datetime format
filtered_data['Lead Creation Date'] = pd.to_datetime(filtered_data['Lead Creation Date'])
filtered_data['Opportunity Order Promised Date'] = pd.to_datetime(filtered_data['Opportunity Order Promised Date'])

# Step 2: Calculate the time difference in days from 'Lead Creation Date' to 'Opportunity Order Promised Date'
filtered_data['Time Difference (Days)'] = (filtered_data['Opportunity Order Promised Date'] - filtered_data['Lead Creation Date']).dt.days

# Step 3: Group by 'Business Unit' and calculate the mean time difference
time_diff_by_business_unit = filtered_data.groupby('Business Unit').agg(
    mean_time_diff=('Time Difference (Days)', 'mean')
).reset_index()

# Step 4: Create a bar chart to visualize the mean time difference across business units
fig_crt_prom = px.bar(
    time_diff_by_business_unit,
    x='Business Unit',
    y='mean_time_diff',
    title="Mean Time Taken from Lead Creation to Order Promised Date by Business Unit",
    labels={'mean_time_diff': 'Mean Time Difference (Days)', 'Business Unit': 'Business Unit'},
    text='mean_time_diff'  # Display the mean time difference on the bars
)

# Customize the layout
fig_crt_prom.update_traces(texttemplate='%{text:.0f} d', textposition='outside')

fig_crt_prom.update_layout(
    yaxis_title="Mean Time Difference (Days)",
    xaxis_title="Business Unit",
    xaxis={'categoryorder': 'total ascending'},
    margin=dict(t=60, b=100, l=40, r=40),
    height=500  # Adjust height if needed
)

# Show the plot
st.plotly_chart(fig_crt_prom)

#%%How often do leads in a specific business category convert?
st.subheader('29. How often do leads in a specific business category convert?')

# Step 1: Define closed stages
closed_stages = ['Order Booked', 'Sales Recognized', 'Order Promised']

# Step 2: Add a column to flag whether the opportunity is closed
filtered_data['Converted'] = filtered_data['StageName'].isin(closed_stages)

# Step 3: Group by 'Business Category' and calculate total leads and total closed opportunities
conversion_data = filtered_data.groupby('Business Category').agg(
    total_leads=('Lead ID', 'size'),  # Count total leads for each Business Category
    total_converted=('Converted', 'sum')  # Count converted leads for each Business Category
).reset_index()

# Step 4: Calculate conversion rate as a percentage
conversion_data['Conversion Rate (%)'] = (conversion_data['total_converted'] / conversion_data['total_leads']) * 100

#remove the ones with zero values
conversion_data = conversion_data[conversion_data['Conversion Rate (%)']>0]

# Step 5: Create a bar chart to visualize the conversion rate by Business Category
fig_biz_conv = px.bar(
    conversion_data,
    x='Business Category',
    y='Conversion Rate (%)',
    #title="Lead-to-Opportunity Conversion Rate by Business Category",
    labels={'Conversion Rate (%)': 'Conversion Rate (%)', 'Business Category': 'Business Category'},
    #text='Conversion Rate (%)'  # Display the conversion rate on the bars
)

# Customize layout
#fig_biz_conv.update_traces(texttemplate='%{text:.1f}%', textposition='outside')

fig_biz_conv.update_layout(
    yaxis_title="Conversion Rate (%)",
    xaxis_title="Business Category",
    xaxis={'categoryorder': 'total descending'},
    margin=dict(t=60, b=100, l=40, r=40),
    height=600,  # Adjust height as needed
    xaxis_tickangle=-45  # Rotate x-axis labels if needed for better readability
)

# Show the plot
st.plotly_chart(fig_biz_conv)

#%%Which user types (e.g., admin, sales) close the most opportunities?
st.subheader('30. Which user types (e.g., admin, sales) close the most opportunities?')

# Step 1: Filter the dataframe for closed opportunities based on 'StageName'
closed_stages = ['Order Booked', 'Sales Recognized', 'Order Promised']
closed_opportunities = filtered_data[filtered_data['StageName'].isin(closed_stages)]

# Step 2: Group by 'Lead Role' and count the number of closed opportunities for each role
role_closed_opportunities = closed_opportunities.groupby('Lead Role').size().reset_index(name='Closed Opportunity Count')

# Step 3: Sort the data by closed opportunity count in descending order for better visualization
role_closed_opportunities = role_closed_opportunities.sort_values(by='Closed Opportunity Count', ascending=False)

# Step 4: Create a bar chart to visualize the user roles contributing the most to closed opportunities
fig_role = px.bar(
    role_closed_opportunities, 
    x='Lead Role', 
    y='Closed Opportunity Count', 
    title="User Roles Contributing to Closed Opportunities",
    labels={'Closed Opportunity Count': 'Closed Opportunity Count', 'Lead Role': 'User Role'},
    text='Closed Opportunity Count'  # Display the closed opportunity count on the bars
)

# Customize the layout
fig_role.update_traces(texttemplate='%{text}', textposition='outside')

fig_role.update_layout(
    yaxis_title="Closed Opportunity Count",
    xaxis_title="User Role",
    xaxis={'categoryorder': 'total descending'},
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    margin=dict(t=60, b=100, l=40, r=40),
    height=600  # Adjust height as needed
)

# Show the plot
st.plotly_chart(fig_role)

#%% Subheader for Q5
st.subheader('5. Revisited: Which campaigns result in the highest lead conversion rates?', divider='rainbow')

# Step 1: Group by 'Campaign Name' and calculate total leads and converted leads
campaign_conversion_data = filtered_data.groupby('Campaign Name').agg(
    total_leads=('Lead ID', 'count'),
    converted_leads=('Opportunity ID', lambda x: x.notna().sum())
).reset_index()

# Step 2: Calculate conversion rate
campaign_conversion_data['conversion_rate'] = (campaign_conversion_data['converted_leads'] / campaign_conversion_data['total_leads']) * 100
campaign_conversion_data['conversion_rate'] = round(campaign_conversion_data['conversion_rate'], 1)

# Step 3: Filter rows where conversion rate > 0
campaign_conversion_data = campaign_conversion_data[campaign_conversion_data['conversion_rate'] > 0]

# Step 4: Sort the data by conversion rate in descending order
campaign_conversion_data = campaign_conversion_data.sort_values(by='converted_leads', ascending=False)

# Step 5: Calculate cumulative contribution for Pareto principle
campaign_conversion_data['Cumulative Contribution'] = campaign_conversion_data['conversion_rate'].cumsum() / campaign_conversion_data['conversion_rate'].sum() * 100

# Step 6: Filter for the top 80% contributors (Pareto view)
pareto_conversion_data = campaign_conversion_data[campaign_conversion_data['Cumulative Contribution'] <= 80]

# Step 7: Group the remaining categories as "Others" for the Pareto view
remaining_categories_conv = campaign_conversion_data[campaign_conversion_data['Cumulative Contribution'] > 80]
others_conv = pd.DataFrame({
    'Campaign Name': ['Others'],
    'total_leads': [remaining_categories_conv['total_leads'].sum()],
    'converted_leads': [remaining_categories_conv['converted_leads'].sum()],
    'conversion_rate': [remaining_categories_conv['conversion_rate'].sum()],
    'Cumulative Contribution': [100.0]  # Since 'Others' will sum up to 100%
})

# Step 8: Create a bar chart to visualize the mean time difference across business units
on_camp = st.toggle("Pareto View", key="on_camp")

if on_camp:
    fig_pareto_conv = px.bar(
    pareto_conversion_data, 
    x='Campaign Name', 
    y='conversion_rate', 
    text='conversion_rate',
    hover_data={'total_leads': True, 'converted_leads': True},  # Show extra info on hover
    title='Lead Conversion Rates by Campaign (Pareto)',
    labels={'Campaign Name': 'Campaign Name', 'conversion_rate': 'Conversion Rate (%)'},
    )

    # Update layout for better display
    fig_pareto_conv.update_traces(texttemplate='%{text:.0f}%', textposition='inside')
    fig_pareto_conv.update_layout(
        xaxis_title='Campaign Name', 
        yaxis_title='Conversion Rate (%)', 
        height=900,
        showlegend=False
    )
    st.plotly_chart(fig_pareto_conv)
else:
    fig_campaign_conversion = px.bar(
        campaign_conversion_data, 
        x='Campaign Name', 
        y='conversion_rate', 
        text='converted_leads', 
        title='Lead Conversion Rates by Campaign',
        labels={'Campaign Name': 'Campaign Name', 'conversion_rate': 'Conversion Rate (%)'},
        color='conversion_rate',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Update layout for better display
    fig_campaign_conversion.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_campaign_conversion.update_layout(
        xaxis_title='Campaign Name', 
        yaxis_title='Conversion Rate (%)', 
        xaxis={'categoryorder':'total descending'},  # Sort the x-axis based on total_leads
        height=900,
        showlegend=False
    )

    st.plotly_chart(fig_campaign_conversion)
