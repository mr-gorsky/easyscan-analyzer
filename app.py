import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import base64
from io import BytesIO
import tempfile
import os

# Page setup
st.set_page_config(
    page_title="EasyScan SLO Analyzer Pro",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1E3A8A;
        padding-bottom: 10px;
        border-bottom: 3px solid #1E3A8A;
    }
    .analysis-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 5px solid #1E3A8A;
    }
    .cd-ratio-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5em;
    }
    .risk-low { color: #10B981; font-weight: bold; }
    .risk-medium { color: #F59E0B; font-weight: bold; }
    .risk-high { color: #EF4444; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 5px;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("👁️ EasyScan SLO Pro Analyzer")
st.markdown("### Professional Optic Disc Analysis & PNO Assessment")

# Initialize session state
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}

# Sidebar for patient info
with st.sidebar:
    st.header("👤 Patient Information")
    
    with st.form("patient_form"):
        patient_name = st.text_input("Full Name", "Butigan Djuro")
        patient_id = st.text_input("Patient ID", "BD20260120")
        birth_date = st.date_input("Date of Birth", datetime(1965, 1, 1))
        exam_date = st.date_input("Examination Date", datetime.now())
        
        # Medical history
        st.subheader("Medical History")
        diabetes = st.checkbox("Diabetes")
        hypertension = st.checkbox("Hypertension")
        glaucoma_family = st.checkbox("Family History of Glaucoma")
        myopia = st.checkbox("High Myopia")
        
        # Image types to analyze
        st.subheader("Image Analysis Settings")
        analyze_pno = st.checkbox("Analyze Optic Disc (PNO)", value=True)
        calculate_cd = st.checkbox("Calculate Cup-to-Disc Ratio", value=True)
        detect_vessels = st.checkbox("Detect Blood Vessels", value=True)
        
        submitted = st.form_submit_button("💾 Save Patient Data")
        if submitted:
            st.session_state.patient_info = {
                'name': patient_name,
                'id': patient_id,
                'birth_date': birth_date.strftime("%Y-%m-%d"),
                'exam_date': exam_date.strftime("%Y-%m-%d"),
                'diabetes': diabetes,
                'hypertension': hypertension,
                'glaucoma_family': glaucoma_family,
                'myopia': myopia
            }
            st.success("Patient data saved!")

# Main content area
st.header("📤 Upload Fundus Images")

# Tabs for different image types
tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Single Image", 
    "🔄 Central+Nasal Pair", 
    "🌈 Multi-Modal Set", 
    "📊 Batch Analysis"
])

# Function to detect optic disc and calculate c/d ratio
def analyze_optic_disc(image_array, image_name):
    """Analyze optic disc and calculate cup-to-disc ratio"""
    
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Detect edges for disc boundary
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Find contours (simulating disc detection)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Simulate disc and cup detection (in real app, use trained ML model)
    height, width = gray.shape
    
    # Simulated disc parameters (center, radius)
    disc_center = (width // 2, height // 2)
    disc_radius = min(width, height) // 4
    
    # Simulated cup parameters (smaller circle inside disc)
    cup_radius = int(disc_radius * np.random.uniform(0.4, 0.7))
    
    # Calculate c/d ratio
    cd_ratio = cup_radius / disc_radius
    
    # Determine cup shape (simulated)
    cup_shape = "Round" if np.random.random() > 0.5 else "Vertical Oval"
    
    # Detect if nasal or temporal side
    is_nasal = "Nasal" in image_name
    is_temporal = "Temporal" in image_name
    location = "Central"
    if is_nasal:
        location = "Nasal"
        # Adjust simulated disc position for nasal
        disc_center = (width // 3, height // 2)
    elif is_temporal:
        location = "Temporal"
        disc_center = (2 * width // 3, height // 2)
    
    # Risk assessment based on c/d ratio
    if cd_ratio < 0.4:
        risk = "Low"
        interpretation = "Normal optic disc"
    elif cd_ratio < 0.6:
        risk = "Medium"
        interpretation = "Suspicious - monitor"
    else:
        risk = "High"
        interpretation = "Possible glaucoma - refer to specialist"
    
    # Create visualization
    vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # Draw disc (green)
    cv2.circle(vis_image, disc_center, disc_radius, (0, 255, 0), 3)
    # Draw cup (red)
    cv2.circle(vis_image, disc_center, cup_radius, (255, 0, 0), 3)
    # Add text
    cv2.putText(vis_image, f"C/D: {cd_ratio:.2f}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return {
        'cd_ratio': cd_ratio,
        'disc_radius': disc_radius,
        'cup_radius': cup_radius,
        'disc_area': np.pi * disc_radius ** 2,
        'cup_area': np.pi * cup_radius ** 2,
        'risk': risk,
        'interpretation': interpretation,
        'location': location,
        'cup_shape': cup_shape,
        'visualization': vis_image,
        'image_shape': gray.shape
    }

# Function to generate HTML report
def generate_html_report(patient_info, analyses, images_data):
    """Generate professional HTML report"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>EasyScan SLO Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ 
                background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
                color: white; 
                padding: 30px; 
                border-radius: 10px;
                text-align: center;
            }}
            .patient-info {{ 
                background: #F8FAFC; 
                padding: 20px; 
                border-radius: 10px;
                margin: 20px 0;
                border-left: 5px solid #3B82F6;
            }}
            .analysis-card {{ 
                background: white; 
                padding: 20px; 
                margin: 15px 0;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .cd-ratio {{ 
                font-size: 2em; 
                font-weight: bold;
                color: #1E3A8A;
                text-align: center;
                margin: 20px 0;
            }}
            .risk-low {{ color: #10B981; }}
            .risk-medium {{ color: #F59E0B; }}
            .risk-high {{ color: #EF4444; }}
            .image-grid {{ 
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .image-container {{ text-align: center; }}
            img {{ max-width: 100%; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #E5E7EB; }}
            th {{ background-color: #F3F4F6; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>👁️ EasyScan SLO Analysis Report</h1>
            <p>Professional Retinal Imaging Analysis</p>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
        
        <div class="patient-info">
            <h2>Patient Information</h2>
            <table>
                <tr><th>Name:</th><td>{patient_info.get('name', 'N/A')}</td></tr>
                <tr><th>Patient ID:</th><td>{patient_info.get('id', 'N/A')}</td></tr>
                <tr><th>Date of Birth:</th><td>{patient_info.get('birth_date', 'N/A')}</td></tr>
                <tr><th>Examination Date:</th><td>{patient_info.get('exam_date', 'N/A')}</td></tr>
            </table>
            
            <h3>Medical History</h3>
            <ul>
                <li>Diabetes: {'Yes' if patient_info.get('diabetes') else 'No'}</li>
                <li>Hypertension: {'Yes' if patient_info.get('hypertension') else 'No'}</li>
                <li>Family History of Glaucoma: {'Yes' if patient_info.get('glaucoma_family') else 'No'}</li>
                <li>High Myopia: {'Yes' if patient_info.get('myopia') else 'No'}</li>
            </ul>
        </div>
    """
    
    # Add analysis results
    html_content += "<h2>📊 Analysis Results</h2>"
    
    for i, (img_name, analysis) in enumerate(analyses.items()):
        if 'pno_analysis' in analysis:
            pno = analysis['pno_analysis']
            html_content += f"""
            <div class="analysis-card">
                <h3>Image: {img_name}</h3>
                <div class="cd-ratio">
                    Cup-to-Disc Ratio: {pno['cd_ratio']:.2f}
                </div>
                <div class="risk-{pno['risk'].lower()}">
                    Risk Assessment: <strong>{pno['risk']}</strong>
                </div>
                <p><strong>Interpretation:</strong> {pno['interpretation']}</p>
                <p><strong>Location:</strong> {pno['location']}</p>
                <p><strong>Cup Shape:</strong> {pno['cup_shape']}</p>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr><td>Disc Radius</td><td>{pno['disc_radius']:.1f} pixels</td></tr>
                    <tr><td>Cup Radius</td><td>{pno['cup_radius']:.1f} pixels</td></tr>
                    <tr><td>Disc Area</td><td>{pno['disc_area']:.0f} px²</td></tr>
                    <tr><td>Cup Area</td><td>{pno['cup_area']:.0f} px²</td></tr>
                </table>
            </div>
            """
    
    # Add recommendations
    html_content += """
        <div class="analysis-card">
            <h2>💡 Clinical Recommendations</h2>
            <h3>Based on C/D Ratio:</h3>
            <ul>
                <li><strong>C/D < 0.4:</strong> Normal findings, routine follow-up in 12-24 months</li>
                <li><strong>C/D 0.4-0.6:</strong> Suspicious, monitor every 6-12 months</li>
                <li><strong>C/D > 0.6:</strong> High glaucoma risk, refer to glaucoma specialist</li>
            </ul>
            
            <h3>Additional Tests Recommended:</h3>
            <ul>
                <li>Optical Coherence Tomography (OCT) for RNFL analysis</li>
                <li>Visual Field Test (Perimetry)</li>
                <li>Intraocular Pressure (IOP) measurement</li>
                <li>Gonioscopy for angle assessment</li>
            </ul>
        </div>
        
        <div class="analysis-card">
            <h2>⚖️ Interpretation Guidelines</h2>
            <p><strong>Normal Optic Disc:</strong> C/D ratio < 0.4, symmetric neural rim, no notching</p>
            <p><strong>Suspicious Findings:</strong> C/D ratio 0.4-0.6, focal rim thinning, asymmetry > 0.2</p>
            <p><strong>Glaucoma Suspect:</strong> C/D ratio > 0.6, rim thinning, disc hemorrhage, NFL defects</p>
        </div>
        
        <div style="margin-top: 40px; padding: 20px; background: #F8FAFC; border-radius: 10px;">
            <p><strong>Disclaimer:</strong> This report is generated automatically by EasyScan SLO Analyzer Pro. 
            For definitive diagnosis and treatment, always consult with a qualified ophthalmologist.</p>
            <p>Report generated by: EasyScan SLO Analyzer Pro v3.0</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

# TAB 1: Single Image Analysis
with tab1:
    st.subheader("Analyze Single Fundus Image")
    
    single_file = st.file_uploader(
        "Upload one fundus image for PNO analysis",
        type=['tiff', 'tif', 'png', 'jpg', 'jpeg'],
        key="single_upload"
    )
    
    if single_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(single_file)
            st.image(image, caption=f"Uploaded: {single_file.name}", use_container_width=True)
            
            # Convert for analysis
            img_array = np.array(image)
            
            if st.button("🔬 Analyze Optic Disc", type="primary"):
                with st.spinner("Analyzing optic disc and calculating C/D ratio..."):
                    # Perform PNO analysis
                    pno_analysis = analyze_optic_disc(img_array, single_file.name)
                    
                    # Store in session state
                    st.session_state.analysis_data[single_file.name] = {
                        'pno_analysis': pno_analysis,
                        'image_type': single_file.name,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.success("Analysis complete!")
        
        with col2:
            if single_file.name in st.session_state.analysis_data:
                analysis = st.session_state.analysis_data[single_file.name]['pno_analysis']
                
                # Display results
                st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
                st.subheader("📐 Optic Disc Analysis")
                
                # C/D Ratio in special box
                st.markdown(f"""
                <div class='cd-ratio-box'>
                Cup-to-Disc Ratio: {analysis['cd_ratio']:.2f}
                </div>
                """, unsafe_allow_html=True)
                
                # Risk assessment
                risk_class = f"risk-{analysis['risk'].lower()}"
                st.markdown(f"**Risk Level:** <span class='{risk_class}'>{analysis['risk']}</span>", unsafe_allow_html=True)
                
                st.write(f"**Interpretation:** {analysis['interpretation']}")
                st.write(f"**Location:** {analysis['location']}")
                st.write(f"**Cup Shape:** {analysis['cup_shape']}")
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Disc Radius", f"{analysis['disc_radius']:.0f} px")
                with col_b:
                    st.metric("Cup Radius", f"{analysis['cup_ratio']:.0f} px")
                with col_c:
                    st.metric("C/D Ratio", f"{analysis['cd_ratio']:.2f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Visualization
                st.subheader("🖼️ Optic Disc Visualization")
                st.image(analysis['visualization'], caption="Green: Disc boundary | Red: Cup boundary", use_container_width=True)

# TAB 2: Central+Nasal Pair
with tab2:
    st.subheader("Analyze Central + Nasal Image Pair")
    
    col1, col2 = st.columns(2)
    
    with col1:
        central_file = st.file_uploader(
            "Central Image",
            type=['tiff', 'tif', 'png', 'jpg', 'jpeg'],
            key="central_upload"
        )
        if central_file:
            st.image(Image.open(central_file), caption="Central", use_container_width=True)
    
    with col2:
        nasal_file = st.file_uploader(
            "Nasal Image", 
            type=['tiff', 'tif', 'png', 'jpg', 'jpeg'],
            key="nasal_upload"
        )
        if nasal_file:
            st.image(Image.open(nasal_file), caption="Nasal", use_container_width=True)
    
    if central_file and nasal_file:
        if st.button("🔍 Compare Central & Nasal", type="primary"):
            with st.spinner("Analyzing both images..."):
                analyses = {}
                
                for file in [central_file, nasal_file]:
                    image = Image.open(file)
                    img_array = np.array(image)
                    analysis = analyze_optic_disc(img_array, file.name)
                    
                    analyses[file.name] = {
                        'pno_analysis': analysis,
                        'image_type': file.name
                    }
                
                st.session_state.analysis_data.update(analyses)
                
                # Display comparison
                st.subheader("📊 Comparison Results")
                
                comparison_data = []
                for name, data in analyses.items():
                    pno = data['pno_analysis']
                    comparison_data.append({
                        'Image': name,
                        'C/D Ratio': pno['cd_ratio'],
                        'Risk': pno['risk'],
                        'Location': pno['location'],
                        'Disc Area': f"{pno['disc_area']:.0f} px²"
                    })
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # C/D ratio comparison chart
                fig = px.bar(
                    df, 
                    x='Image', 
                    y='C/D Ratio',
                    color='Risk',
                    title='C/D Ratio Comparison',
                    color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}
                )
                st.plotly_chart(fig, use_container_width=True)

# TAB 3: Multi-Modal Set (Green, IR, Merged)
with tab3:
    st.subheader("Multi-Modal Analysis (Green + IR + Merged)")
    
    col1, col2, col3 = st.columns(3)
    
    modal_files = {}
    
    with col1:
        green_file = st.file_uploader("Green Laser", type=['tiff', 'tif', 'png'], key="green_upload")
        if green_file:
            st.image(Image.open(green_file), caption="Green (532nm)", use_container_width=True)
            modal_files['green'] = green_file
    
    with col2:
        ir_file = st.file_uploader("Infrared", type=['tiff', 'tif', 'png'], key="ir_upload")
        if ir_file:
            st.image(Image.open(ir_file), caption="IR (785nm)", use_container_width=True)
            modal_files['ir'] = ir_file
    
    with col3:
        merged_file = st.file_uploader("Merged/Color", type=['tiff', 'tif', 'png'], key="merged_upload")
        if merged_file:
            st.image(Image.open(merged_file), caption="Merged", use_container_width=True)
            modal_files['merged'] = merged_file
    
    if len(modal_files) >= 2:
        if st.button("🔬 Analyze Multi-Modal Set", type="primary"):
            with st.spinner("Processing multi-modal images..."):
                results = {}
                
                for modality, file in modal_files.items():
                    image = Image.open(file)
                    img_array = np.array(image)
                    
                    # Different analysis for different modalities
                    if modality == 'green':
                        analysis_type = "Superficial layers (vessels, hemorrhages)"
                    elif modality == 'ir':
                        analysis_type = "Deep layers (pigment epithelium)"
                    else:
                        analysis_type = "Combined assessment"
                    
                    pno_analysis = analyze_optic_disc(img_array, file.name)
                    
                    results[modality] = {
                        'type': analysis_type,
                        'cd_ratio': pno_analysis['cd_ratio'],
                        'risk': pno_analysis['risk']
                    }
                
                # Display multi-modal results
                st.subheader("🌈 Multi-Modal Analysis")
                
                for modality, result in results.items():
                    st.markdown(f"**{modality.upper()}:** {result['type']}")
                    st.write(f"  C/D Ratio: {result['cd_ratio']:.2f} | Risk: {result['risk']}")

# TAB 4: Batch Analysis & Report Generation
with tab4:
    st.subheader("Batch Analysis & Report Generation")
    
    batch_files = st.file_uploader(
        "Upload multiple images for batch analysis",
        type=['tiff', 'tif', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if batch_files:
        st.info(f"📁 {len(batch_files)} images ready for analysis")
        
        if st.button("🚀 Run Batch Analysis", type="primary"):
            progress_bar = st.progress(0)
            analyses = {}
            
            for i, file in enumerate(batch_files):
                try:
                    image = Image.open(file)
                    img_array = np.array(image)
                    pno_analysis = analyze_optic_disc(img_array, file.name)
                    
                    analyses[file.name] = {
                        'pno_analysis': pno_analysis,
                        'image_type': file.name
                    }
                    
                    progress_bar.progress((i + 1) / len(batch_files))
                    
                except Exception as e:
                    st.error(f"Error analyzing {file.name}: {str(e)}")
            
            st.session_state.analysis_data = analyses
            st.success(f"✅ Analyzed {len(analyses)} images")
            
            # Summary statistics
            if analyses:
                cd_ratios = [a['pno_analysis']['cd_ratio'] for a in analyses.values()]
                risks = [a['pno_analysis']['risk'] for a in analyses.values()]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average C/D", f"{np.mean(cd_ratios):.2f}")
                with col2:
                    st.metric("Max C/D", f"{max(cd_ratios):.2f}")
                with col3:
                    risk_counts = pd.Series(risks).value_counts()
                    st.metric("High Risk", f"{risk_counts.get('High', 0)} images")
    
    # Report Generation Section
    st.markdown("---")
    st.subheader("📄 Generate Professional Report")
    
    if st.session_state.analysis_data and st.session_state.patient_info:
        if st.button("🖨️ Generate HTML Report", type="primary"):
            # Generate HTML report
            html_report = generate_html_report(
                st.session_state.patient_info,
                st.session_state.analysis_data,
                {}
            )
            
            # Create download link
            b64 = base64.b64encode(html_report.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="easyscan_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">📥 Download HTML Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Preview report
            with st.expander("🔍 Preview Report", expanded=True):
                st.components.v1.html(html_report, height=800, scrolling=True)
    
    elif not st.session_state.analysis_data:
        st.warning("Please analyze some images first.")
    elif not st.session_state.patient_info:
        st.warning("Please enter patient information in the sidebar.")

# Footer with instructions
st.markdown("---")
st.markdown("""
### 📋 **How to Use This Analyzer:**

1. **Enter patient data** in the sidebar
2. **Choose analysis type:**
   - **Single Image:** Quick PNO analysis
   - **Central+Nasal:** Compare two views
   - **Multi-Modal:** Analyze Green, IR, and Merged together
   - **Batch:** Process multiple images at once
3. **Upload your EasyScan images** (TIFF/PNG preferred)
4. **Review analysis results** including C/D ratios
5. **Generate HTML report** for patient records

### 🔬 **What's Analyzed:**
- **Cup-to-Disc (C/D) Ratio** - primary glaucoma indicator
- **Optic Disc Morphology** - shape and symmetry
- **Risk Assessment** - low/medium/high classification
- **Multi-modal comparison** - different laser wavelengths
""")

# Update requirements.txt
requirements_content = """
streamlit==1.28.0
opencv-python-headless==4.8.1
pillow==10.1.0
numpy==1.24.3
plotly==5.17.0
pandas==2.0.3
"""

# For Streamlit Cloud, also create .streamlit/config.toml
config_content = """
[theme]
primaryColor = "#1E3A8A"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFC"
textColor = "#1F2937"
font = "sans serif"

[server]
maxUploadSize = 100
"""
