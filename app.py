import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="GENESIS - AI Strategy Solutions",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=GENESIS", width=200)
    st.markdown("---")
    
    mode = st.selectbox(
        "🎯 Choose Mode",
        ["Chat Assistant", "Report Generation", "Analytics Dashboard"]
    )
    
    st.markdown("---")
    st.markdown("### 🔧 Settings")
    
    if mode == "Chat Assistant":
        model = st.selectbox("Model", ["GPT-4 Turbo", "GPT-4", "GPT-3.5 Turbo"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    elif mode == "Report Generation":
        industry = st.selectbox("Industry", ["Fintech", "Healthcare", "E-commerce", "Manufacturing", "Education"])
        report_type = st.selectbox("Report Type", ["Market Analysis", "Use Case Study", "Competitive Analysis"])
    
    st.markdown("---")
    st.markdown("### 📊 Recent Activity")
    st.metric("Reports Generated", "47", "↗️ 12%")
    st.metric("Chats Completed", "156", "↗️ 8%")
    st.metric("Data Sources", "23", "↗️ 3%")

# Main content
st.markdown('<h1 class="main-header">🚀 GENESIS</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Generative AI for Strategy and Industry Solutions</p>', unsafe_allow_html=True)

if mode == "Chat Assistant":
    st.markdown("## 💬 AI Strategy Assistant")
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat history
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-message user-message"><strong>You:</strong> What are the top AI use cases in fintech?</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-message bot-message"><strong>GENESIS:</strong> Based on my analysis of current fintech trends, here are the top AI use cases:<br><br>1. <strong>Fraud Detection</strong> - Real-time transaction monitoring<br>2. <strong>Credit Scoring</strong> - Alternative data assessment<br>3. <strong>Robo-Advisory</strong> - Automated investment management<br>4. <strong>Chatbots</strong> - Customer service automation<br>5. <strong>Regulatory Compliance</strong> - Automated reporting</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-message user-message"><strong>You:</strong> Can you elaborate on fraud detection implementation?</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-message bot-message"><strong>GENESIS:</strong> Certainly! Fraud detection implementation typically involves:<br><br>🔍 <strong>Data Collection:</strong> Transaction patterns, user behavior, device fingerprinting<br>🤖 <strong>ML Models:</strong> Anomaly detection, neural networks, ensemble methods<br>⚡ <strong>Real-time Processing:</strong> Stream processing for instant decisions<br>📊 <strong>Risk Scoring:</strong> Dynamic risk assessment algorithms</div>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Ask me anything about AI strategy...", placeholder="e.g., How can AI improve customer retention in e-commerce?")
        
        col_send, col_clear = st.columns([1, 1])
        with col_send:
            if st.button("Send 📤", use_container_width=True):
                st.success("Message sent! Processing...")
        with col_clear:
            if st.button("Clear Chat 🗑️", use_container_width=True):
                st.success("Chat cleared!")
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        st.metric("Response Time", "1.2s", "↓ 0.3s")
        st.metric("Accuracy", "94%", "↗️ 2%")
        st.metric("Sources Used", "12", "↗️ 4")
        
        st.markdown("### 🎯 Suggested Topics")
        if st.button("💰 Fintech Trends", use_container_width=True):
            st.info("Loading fintech insights...")
        if st.button("🏥 Healthcare AI", use_container_width=True):
            st.info("Loading healthcare analysis...")
        if st.button("🛒 E-commerce Solutions", use_container_width=True):
            st.info("Loading e-commerce data...")

elif mode == "Report Generation":
    st.markdown("## 📊 AI Strategy Report Generator")
    
    # Report generation interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Report Configuration")
        
        company_name = st.text_input("Company Name", placeholder="e.g., TechCorp Solutions")
        
        col_ind, col_size = st.columns(2)
        with col_ind:
            selected_industry = st.selectbox("Industry Sector", ["Fintech", "Healthcare", "E-commerce", "Manufacturing", "Education", "Retail", "Insurance"])
        with col_size:
            company_size = st.selectbox("Company Size", ["Startup", "Small (< 50)", "Medium (50-500)", "Large (500+)", "Enterprise"])
        
        focus_areas = st.multiselect(
            "Focus Areas",
            ["Customer Service", "Operations", "Marketing", "Sales", "HR", "Finance", "R&D", "Supply Chain"],
            default=["Customer Service", "Operations"]
        )
        
        col_depth, col_format = st.columns(2)
        with col_depth:
            report_depth = st.select_slider("Report Depth", ["Basic", "Standard", "Comprehensive", "Enterprise"], value="Standard")
        with col_format:
            output_format = st.selectbox("Output Format", ["PDF", "DOCX", "HTML", "PowerPoint"])
        
        # Generate button
        if st.button("🚀 Generate Report", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 20:
                    status_text.text("🔍 Scraping industry data...")
                elif i < 40:
                    status_text.text("📚 Processing documents...")
                elif i < 60:
                    status_text.text("🤖 Analyzing with GPT-4...")
                elif i < 80:
                    status_text.text("📊 Generating insights...")
                else:
                    status_text.text("📄 Compiling report...")
                time.sleep(0.02)
            
            status_text.text("✅ Report generated successfully!")
            st.success("Report ready for download!")
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data="Sample report content...",
                file_name=f"{company_name}_AI_Strategy_Report.{output_format.lower()}",
                mime="application/octet-stream"
            )
    
    with col2:
        st.markdown("### 📋 Report Preview")
        
        # Sample report structure
        st.markdown("""
        **Executive Summary**
        - Market Overview
        - Key Opportunities
        - Strategic Recommendations
        
        **Industry Analysis**
        - Current AI Adoption
        - Competitive Landscape
        - Emerging Trends
        
        **Use Case Recommendations**
        - Priority Use Cases
        - Implementation Roadmap
        - ROI Projections
        
        **Technical Requirements**
        - Infrastructure Needs
        - Skills Assessment
        - Technology Stack
        
        **Implementation Plan**
        - Timeline & Milestones
        - Resource Allocation
        - Risk Assessment
        """)
        
        st.markdown("### 🎯 Sample Insights")
        st.info("💡 AI chatbots can reduce customer service costs by 30-50%")
        st.info("📈 Predictive analytics can improve demand forecasting by 25%")
        st.info("🔒 AI-powered fraud detection reduces false positives by 60%")

else:  # Analytics Dashboard
    st.markdown("## 📊 Analytics Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reports", "1,247", "↗️ 15%")
    with col2:
        st.metric("Active Users", "89", "↗️ 7%")
    with col3:
        st.metric("Data Sources", "156", "↗️ 12%")
    with col4:
        st.metric("API Calls", "24.7K", "↗️ 23%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Report Generation Trends")
        
        # Sample data for chart
        dates = pd.date_range(start='2024-01-01', end='2024-06-18', freq='D')
        data = pd.DataFrame({
            'Date': dates,
            'Reports': [20 + i % 30 + (i // 7) * 2 for i in range(len(dates))]
        })
        
        fig = px.line(data, x='Date', y='Reports', title="Daily Report Generation")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏭 Industry Distribution")
        
        industry_data = pd.DataFrame({
            'Industry': ['Fintech', 'Healthcare', 'E-commerce', 'Manufacturing', 'Education', 'Other'],
            'Count': [145, 98, 87, 76, 54, 89]
        })
        
        fig = px.pie(industry_data, values='Count', names='Industry', title="Reports by Industry")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.markdown("### 📋 Recent Activity")
    
    activity_data = pd.DataFrame({
        'Time': ['2 min ago', '5 min ago', '12 min ago', '18 min ago', '25 min ago'],
        'User': ['john.doe@company.com', 'sarah.smith@startup.io', 'mike.wilson@corp.com', 'lisa.chen@tech.co', 'david.brown@firm.com'],
        'Action': ['Generated Healthcare Report', 'Started Chat Session', 'Downloaded Fintech Analysis', 'Completed E-commerce Study', 'Uploaded Custom Data'],
        'Status': ['✅ Completed', '🔄 In Progress', '✅ Completed', '✅ Completed', '✅ Completed']
    })
    
    st.dataframe(activity_data, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">🚀 GENESIS v1.0 | Powered by GPT-4 & Advanced Analytics</p>',
    unsafe_allow_html=True
)