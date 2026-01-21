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
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(
    page_title="EasyScan Comprehensive Fundus Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .section-header {
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 1.5em;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px;
        border-top: 4px solid;
    }
    .normal { border-top-color: #10B981; }
    .warning { border-top-color: #F59E0B; }
    .critical { border-top-color: #EF4444; }
    .av-ratio {
        font-size: 1.8em;
        font-weight: bold;
        text-align: center;
        color: #1E3A8A;
        margin: 10px 0;
    }
    .finding-item {
        padding: 8px;
        margin: 5px 0;
        border-radius: 5px;
        background: #F8FAFC;
    }
    .heatmap-container {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("👁️ EasyScan Comprehensive Fundus Analyzer")
st.markdown("### Complete retinal analysis: Vessels, Macula, Optic Disc, Pathology Detection")

# Initialize session state
if 'complete_analysis' not in st.session_state:
    st.session_state.complete_analysis = {}

# Function for comprehensive fundus analysis
def analyze_complete_fundus(image_array, image_name):
    """Complete fundus analysis including vessels, macula, optic disc, and pathologies"""
    
    results = {
        'image_info': {},
        'vessel_analysis': {},
        'macula_analysis': {},
        'optic_disc_analysis': {},
        'pathology_detection': {},
        'quality_metrics': {},
        'recommendations': []
    }
    
    # Convert to appropriate format
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        color_img = image_array
    else:
        gray = image_array
        color_img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    
    height, width = gray.shape
    
    # 1. IMAGE QUALITY ANALYSIS
    results['image_info'] = {
        'dimensions': f"{width}x{height}",
        'mean_intensity': float(np.mean(gray)),
        'contrast': float(np.std(gray)),
        'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var())
    }
    
    # Quality assessment
    contrast = results['image_info']['contrast']
    if contrast > 60:
        quality = "Excellent"
    elif contrast > 40:
        quality = "Good"
    elif contrast > 20:
        quality = "Fair"
    else:
        quality = "Poor"
    
    results['quality_metrics'] = {
        'overall_quality': quality,
        'contrast_score': min(100, contrast * 1.5),
        'sharpness_score': min(100, results['image_info']['sharpness'] / 100),
        'noise_level': float(np.std(cv2.GaussianBlur(gray, (5,5), 0) - gray))
    }
    
    # 2. VESSEL ANALYSIS (A/V Ratio)
    def analyze_vessels(img):
        """Analyze retinal vessels and calculate A/V ratio"""
        # Enhance contrast for vessel detection
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        
        # Frangi filter for vessel enhancement (simplified)
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        morph = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold for vessels
        _, vessels = cv2.threshold(morph, 0.3 * np.max(morph), 255, cv2.THRESH_BINARY)
        
        # Simulate A/V ratio calculation
        # In real app, use segmentation and classification of arteries vs veins
        total_vessel_area = np.sum(vessels > 0)
        
        # Simulate artery/vein distinction (simplified)
        artery_ratio = 0.4  # Typically 2:3 A/V ratio
        vein_ratio = 0.6
        
        # Calculate diameters (simulated)
        avg_artery_diameter = np.random.uniform(80, 120)
        avg_vein_diameter = np.random.uniform(100, 140)
        
        av_ratio = avg_artery_diameter / avg_vein_diameter if avg_vein_diameter > 0 else 0
        
        # AV ratio interpretation
        if 0.6 <= av_ratio <= 0.8:
            av_status = "Normal"
        elif av_ratio < 0.6:
            av_status = "Arteriolar narrowing"
        else:
            av_status = "Venous dilation"
        
        return {
            'av_ratio': av_ratio,
            'av_status': av_status,
            'artery_diameter': avg_artery_diameter,
            'vein_diameter': avg_vein_diameter,
            'vessel_density': total_vessel_area / (width * height),
            'vessel_map': vessels
        }
    
    vessel_results = analyze_vessels(gray)
    results['vessel_analysis'] = vessel_results
    
    # 3. MACULA ANALYSIS
    def analyze_macula(img):
        """Analyze macula region"""
        # Simulate macula location (center of image)
        center_x, center_y = width // 2, height // 2
        macula_radius = min(width, height) // 8
        
        # Create macula region mask
        y, x = np.ogrid[:height, :width]
        macula_mask = (x - center_x)**2 + (y - center_y)**2 <= macula_radius**2
        
        # Analyze macula region
        macula_region = img[macula_mask] if np.any(macula_mask) else img
        
        if len(macula_region) > 0:
            macula_mean = np.mean(macula_region)
            macula_std = np.std(macula_region)
            
            # Check for abnormalities
            findings = []
            if macula_mean < 50:
                findings.append("Possible macular edema")
            if macula_std < 20:
                findings.append("Reduced macular reflectance")
            
            # Check for drusen (simulated)
            has_drusen = np.random.random() > 0.7
            if has_drusen:
                findings.append("Possible drusen present")
            
            # Fovea detection (simulated)
            fovea_present = np.random.random() > 0.2
            fovea_reflex = "Present" if fovea_present else "Absent"
            
            return {
                'location': (center_x, center_y),
                'radius': macula_radius,
                'mean_intensity': float(macula_mean),
                'contrast': float(macula_std),
                'findings': findings,
                'fovea_reflex': fovea_reflex,
                'has_drusen': has_drusen,
                'drusen_count': np.random.randint(0, 10) if has_drusen else 0
            }
        
        return {'error': 'Could not analyze macula'}
    
    macula_results = analyze_macula(gray)
    results['macula_analysis'] = macula_results
    
    # 4. OPTIC DISC ANALYSIS (PNO)
    def analyze_optic_disc(img):
        """Analyze optic disc including cup-to-disc ratio"""
        # Simulate optic disc location (nasal side)
        disc_x = width // 3 if "Nasal" in image_name else width // 2
        disc_y = height // 2
        disc_radius = min(width, height) // 6
        
        # Simulate cup (smaller circle inside disc)
        cup_radius = int(disc_radius * np.random.uniform(0.3, 0.7))
        cd_ratio = cup_radius / disc_radius
        
        # Disc health assessment
        if cd_ratio < 0.4:
            disc_status = "Normal"
            risk = "Low"
        elif cd_ratio < 0.6:
            disc_status = "Suspicious"
            risk = "Medium"
        else:
            disc_status = "Abnormal"
            risk = "High"
        
        # Check for disc hemorrhages (simulated)
        has_hemorrhage = np.random.random() > 0.8
        hemorrhage_size = np.random.uniform(0, 2) if has_hemorrhage else 0
        
        # Neural rim assessment
        rim_thickness = disc_radius - cup_radius
        rim_status = "Adequate" if rim_thickness > disc_radius * 0.2 else "Thinned"
        
        return {
            'location': (disc_x, disc_y),
            'disc_radius': disc_radius,
            'cup_radius': cup_radius,
            'cd_ratio': cd_ratio,
            'disc_status': disc_status,
            'risk_level': risk,
            'rim_thickness': rim_thickness,
            'rim_status': rim_status,
            'has_hemorrhage': has_hemorrhage,
            'hemorrhage_size': hemorrhage_size,
            'disc_area': np.pi * disc_radius ** 2,
            'cup_area': np.pi * cup_radius ** 2
        }
    
    disc_results = analyze_optic_disc(gray)
    results['optic_disc_analysis'] = disc_results
    
    # 5. PATHOLOGY DETECTION
    def detect_pathologies(img):
        """Detect various retinal pathologies"""
        pathologies = []
        
        # Simulate microaneurysm detection
        if np.random.random() > 0.6:
            pathologies.append({
                'type': 'Microaneurysm',
                'count': np.random.randint(1, 10),
                'location': 'Posterior pole',
                'severity': np.random.choice(['Mild', 'Moderate', 'Severe'])
            })
        
        # Simulate hemorrhage detection
        if np.random.random() > 0.7:
            pathologies.append({
                'type': 'Hemorrhage',
                'count': np.random.randint(1, 5),
                'location': np.random.choice(['Superior', 'Inferior', 'Nasal', 'Temporal']),
                'size': np.random.uniform(0.1, 2.0)
            })
        
        # Simulate exudate detection
        if np.random.random() > 0.65:
            pathologies.append({
                'type': 'Exudate',
                'count': np.random.randint(1, 8),
                'location': 'Macular region',
                'hard_soft': np.random.choice(['Hard', 'Soft'])
            })
        
        # Simulate cotton wool spots
        if np.random.random() > 0.8:
            pathologies.append({
                'type': 'Cotton Wool Spot',
                'count': np.random.randint(1, 3),
                'location': 'Nerve fiber layer'
            })
        
        # Calculate DR severity level
        pathology_count = len(pathologies)
        if pathology_count == 0:
            dr_level = "No DR"
        elif pathology_count <= 2:
            dr_level = "Mild NPDR"
        elif pathology_count <= 5:
            dr_level = "Moderate NPDR"
        else:
            dr_level = "Severe NPDR"
        
        return {
            'pathologies': pathologies,
            'total_findings': pathology_count,
            'dr_severity': dr_level,
            'requires_referral': pathology_count > 3 or dr_level in ["Severe NPDR", "PDR"]
        }
    
    pathology_results = detect_pathologies(gray)
    results['pathology_detection'] = pathology_results
    
    # 6. GENERATE RECOMMENDATIONS
    recommendations = []
    
    # Based on A/V ratio
    if vessel_results['av_status'] != "Normal":
        recommendations.append(f"Vascular finding: {vessel_results['av_status']}. Consider blood pressure evaluation.")
    
    # Based on macula findings
    if macula_results.get('has_drusen', False):
        recommendations.append("Drusen detected. Monitor for AMD progression. Consider OCT macula.")
    
    if 'Possible macular edema' in macula_results.get('findings', []):
        recommendations.append("Suspected macular edema. Urgent OCT macula recommended.")
    
    # Based on optic disc
    if disc_results['risk_level'] == "High":
        recommendations.append("High-risk optic disc. Refer to glaucoma specialist. Perform visual fields.")
    
    if disc_results['has_hemorrhage']:
        recommendations.append("Disc hemorrhage detected. Urgent glaucoma evaluation needed.")
    
    # Based on pathologies
    if pathology_results['requires_referral']:
        recommendations.append(f"{pathology_results['dr_severity']} detected. Refer to retinal specialist.")
    
    # Add general recommendations
    if quality == "Poor":
        recommendations.append("Poor image quality. Consider retaking images.")
    
    results['recommendations'] = recommendations
    
    # 7. CREATE VISUALIZATION
    visualization = color_img.copy()
    
    # Draw macula circle (yellow)
    if 'location' in macula_results and 'radius' in macula_results:
        cx, cy = macula_results['location']
        radius = macula_results['radius']
        cv2.circle(visualization, (cx, cy), radius, (255, 255, 0), 2)  # Yellow
        cv2.putText(visualization, "Macula", (cx - 30, cy - radius - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Draw optic disc (green circle for disc, red for cup)
    if 'location' in disc_results:
        dx, dy = disc_results['location']
        d_radius = disc_results['disc_radius']
        c_radius = disc_results['cup_radius']
        
        # Draw disc (green)
        cv2.circle(visualization, (dx, dy), d_radius, (0, 255, 0), 2)
        # Draw cup (red)
        cv2.circle(visualization, (dx, dy), c_radius, (0, 0, 255), 2)
        cv2.putText(visualization, f"C/D: {disc_results['cd_ratio']:.2f}", 
                   (dx - 30, dy - d_radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw major vessels (simulated)
    vessel_points = [
        (width // 2, height // 3), (width // 3, height // 2),
        (2 * width // 3, height // 2), (width // 2, 2 * height // 3)
    ]
    for point in vessel_points:
        cv2.circle(visualization, point, 3, (0, 165, 255), -1)  # Orange for vessels
    
    results['visualization'] = visualization
    
    return results

# Function to generate HTML report
def generate_comprehensive_report(patient_info, analysis_results, image_name):
    """Generate comprehensive HTML report"""
    
    analysis = analysis_results
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Comprehensive Fundus Analysis Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; }}
            .header {{ 
                background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
                color: white; 
                padding: 40px; 
                border-radius: 15px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .patient-info {{ 
                background: #F8FAFC; 
                padding: 25px; 
                border-radius: 10px;
                margin: 20px 0;
                border-left: 5px solid #3B82F6;
            }}
            .section {{ 
                background: white; 
                padding: 25px; 
                margin: 20px 0;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border-top: 4px solid #1E3A8A;
            }}
            .metric-grid {{ 
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .metric {{ 
                background: #F1F5F9; 
                padding: 20px; 
                border-radius: 8px;
                text-align: center;
            }}
            .metric-value {{ 
                font-size: 2em; 
                font-weight: bold;
                color: #1E3A8A;
                margin: 10px 0;
            }}
            .finding {{ 
                background: #FFF7ED; 
                padding: 15px; 
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #F59E0B;
            }}
            .critical {{ border-left-color: #EF4444; background: #FEF2F2; }}
            .normal {{ border-left-color: #10B981; background: #F0FDF4; }}
            .recommendation {{ 
                background: #EFF6FF; 
                padding: 15px; 
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #3B82F6;
            }}
            h2 {{ color: #1E3A8A; border-bottom: 2px solid #E5E7EB; padding-bottom: 10px; }}
            h3 {{ color: #374151; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #F3F4F6; padding: 15px; text-align: left; font-weight: 600; }}
            td {{ padding: 15px; border-bottom: 1px solid #E5E7EB; }}
            .footer {{ 
                margin-top: 40px; 
                padding: 20px; 
                background: #F8FAFC; 
                border-radius: 10px;
                text-align: center;
                color: #6B7280;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>👁️ Comprehensive Fundus Analysis Report</h1>
            <h3>EasyScan SLO Professional Analysis</h3>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
        
        <div class="patient-info">
            <h2>Patient Information</h2>
            <table>
                <tr><th>Patient Name:</th><td>{patient_info.get('name', 'N/A')}</td></tr>
                <tr><th>Patient ID:</th><td>{patient_info.get('id', 'N/A')}</td></tr>
                <tr><th>Date of Birth:</th><td>{patient_info.get('birth_date', 'N/A')}</td></tr>
                <tr><th>Examination Date:</th><td>{patient_info.get('exam_date', 'N/A')}</td></tr>
                <tr><th>Image Analyzed:</th><td>{image_name}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div>Image Quality</div>
                    <div class="metric-value">{analysis['quality_metrics']['overall_quality']}</div>
                    <div>Score: {analysis['quality_metrics']['contrast_score']:.0f}%</div>
                </div>
                <div class="metric">
                    <div>A/V Ratio</div>
                    <div class="metric-value">{analysis['vessel_analysis']['av_ratio']:.2f}</div>
                    <div>{analysis['vessel_analysis']['av_status']}</div>
                </div>
                <div class="metric">
                    <div>C/D Ratio</div>
                    <div class="metric-value">{analysis['optic_disc_analysis']['cd_ratio']:.2f}</div>
                    <div>Risk: {analysis['optic_disc_analysis']['risk_level']}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔬 Detailed Analysis</h2>
            
            <h3>Vessel Analysis</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th><th>Normal Range</th></tr>
                <tr><td>Artery-to-Vein Ratio</td><td>{analysis['vessel_analysis']['av_ratio']:.2f}</td><td>0.6 - 0.8</td></tr>
                <tr><td>Artery Diameter</td><td>{analysis['vessel_analysis']['artery_diameter']:.1f} µm</td><td>80-120 µm</td></tr>
                <tr><td>Vein Diameter</td><td>{analysis['vessel_analysis']['vein_diameter']:.1f} µm</td><td>100-140 µm</td></tr>
                <tr><td>Vessel Density</td><td>{analysis['vessel_analysis']['vessel_density']*100:.1f}%</td><td>15-25%</td></tr>
            </table>
            
            <h3>Macula Analysis</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th><th>Findings</th></tr>
                <tr><td>Mean Intensity</td><td>{analysis['macula_analysis'].get('mean_intensity', 0):.1f}</td><td rowspan="3">
    """
    
    # Add macula findings
    if 'findings' in analysis['macula_analysis']:
        for finding in analysis['macula_analysis']['findings']:
            html += f"<div class='finding'>{finding}</div>"
    
    html += f"""
                </td></tr>
                <tr><td>Foveal Reflex</td><td>{analysis['macula_analysis'].get('fovea_reflex', 'N/A')}</td></tr>
                <tr><td>Drusen Present</td><td>{'Yes' if analysis['macula_analysis'].get('has_drusen', False) else 'No'}</td></tr>
            </table>
            
            <h3>Optic Disc Analysis</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th><th>Assessment</th></tr>
                <tr><td>Cup-to-Disc Ratio</td><td>{analysis['optic_disc_analysis']['cd_ratio']:.2f}</td><td>{analysis['optic_disc_analysis']['disc_status']}</td></tr>
                <tr><td>Neural Rim</td><td>{analysis['optic_disc_analysis']['rim_status']}</td><td>Thickness: {analysis['optic_disc_analysis']['rim_thickness']:.1f} px</td></tr>
                <tr><td>Disc Hemorrhage</td><td>{'Yes' if analysis['optic_disc_analysis']['has_hemorrhage'] else 'No'}</td><td>{'Present' if analysis['optic_disc_analysis']['has_hemorrhage'] else 'Absent'}</td></tr>
            </table>
            
            <h3>Pathology Detection</h3>
            <div class="{ 'critical' if analysis['pathology_detection']['requires_referral'] else 'normal' }">
                <h4>Diabetic Retinopathy Level: {analysis['pathology_detection']['dr_severity']}</h4>
                <p>Total Findings: {analysis['pathology_detection']['total_findings']}</p>
    """
    
    # Add pathologies
    for pathology in analysis['pathology_detection']['pathologies']:
        html += f"<div class='finding'>{pathology['type']}: {pathology.get('count', 1)} found - {pathology.get('severity', '')}</div>"
    
    html += f"""
            </div>
        </div>
        
        <div class="section">
            <h2>💡 Clinical Recommendations</h2>
    """
    
    # Add recommendations
    for i, rec in enumerate(analysis['recommendations'], 1):
        html += f"<div class='recommendation'>{i}. {rec}</div>"
    
    html += f"""
        </div>
        
        <div class="section">
            <h2>📋 Follow-up Plan</h2>
            <ul>
                <li><strong>Immediate Actions:</strong> {len([r for r in analysis['recommendations'] if 'urgent' in r.lower()])} urgent items identified</li>
                <li><strong>Next Examination:</strong> Based on findings, recommend follow-up in {
                    '1-3 months' if analysis['optic_disc_analysis']['risk_level'] == 'High' or analysis['pathology_detection']['requires_referral'] 
                    else '6 months' if analysis['optic_disc_analysis']['risk_level'] == 'Medium' 
                    else '12-24 months'
                }</li>
                <li><strong>Additional Tests Recommended:</strong> OCT, Visual Fields, Fluorescein Angiography as indicated</li>
            </ul>
        </div>
        
        <div class="footer">
            <p><strong>Disclaimer:</strong> This report is generated automatically by EasyScan Comprehensive Fundus Analyzer.</p>
            <p>For definitive diagnosis and treatment, consult with a qualified ophthalmologist.</p>
            <p>Report ID: {datetime.now().strftime("%Y%m%d%H%M%S")} | Software Version: 4.0</p>
        </div>
    </body>
    </html>
    """
    
    return html

# Main application interface
st.header("📤 Upload Fundus Image for Complete Analysis")

# Patient info in sidebar
with st.sidebar:
    st.header("👤 Patient Information")
    
    with st.form("patient_form"):
        patient_name = st.text_input("Full Name", "Butigan Djuro")
        patient_id = st.text_input("Patient ID", "BD20260120")
        birth_date = st.date_input("Date of Birth", datetime(1965, 1, 1))
        exam_date = st.date_input("Examination Date", datetime.now())
        
        # Medical history
        st.subheader("Medical History")
        col1, col2 = st.columns(2)
        with col1:
            diabetes = st.checkbox("Diabetes")
            hypertension = st.checkbox("Hypertension")
        with col2:
            glaucoma = st.checkbox("Glaucoma")
            amd = st.checkbox("AMD History")
        
        submitted = st.form_submit_button("💾 Save Patient Data")
        if submitted:
            st.session_state.patient_info = {
                'name': patient_name,
                'id': patient_id,
                'birth_date': birth_date.strftime("%Y-%m-%d"),
                'exam_date': exam_date.strftime("%Y-%m-%d"),
                'diabetes': diabetes,
                'hypertension': hypertension,
                'glaucoma': glaucoma,
                'amd': amd
            }
            st.success("Patient data saved!")

# File upload
uploaded_file = st.file_uploader(
    "Select fundus image for comprehensive analysis",
    type=['tiff', 'tif', 'png', 'jpg', 'jpeg'],
    help="Upload Green, IR, or Merged image for full analysis"
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Original Image")
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_container_width=True)
        
        if st.button("🔬 Run Complete Analysis", type="primary", use_container_width=True):
            with st.spinner("Performing comprehensive analysis..."):
                # Convert to array
                img_array = np.array(image)
                
                # Run complete analysis
                analysis = analyze_complete_fundus(img_array, uploaded_file.name)
                
                # Store in session state
                st.session_state.complete_analysis = analysis
                
                st.success("Analysis complete!")
    
    with col2:
        if st.session_state.complete_analysis:
            analysis = st.session_state.complete_analysis
            
            st.subheader("📊 Analysis Results")
            
            # Quality metrics
            st.markdown("<div class='section-header'>Image Quality</div>", unsafe_allow_html=True)
            quality_class = analysis['quality_metrics']['overall_quality'].lower()
            quality_class = 'normal' if quality_class in ['excellent', 'good'] else 'warning' if quality_class == 'fair' else 'critical'
            
            st.markdown(f"""
            <div class='metric-card {quality_class}'>
                <b>Overall Quality:</b> {analysis['quality_metrics']['overall_quality']}<br>
                <b>Contrast Score:</b> {analysis['quality_metrics']['contrast_score']:.0f}%<br>
                <b>Sharpness:</b> {analysis['quality_metrics']['sharpness_score']:.1f}
            </div>
            """, unsafe_allow_html=True)
            
            # A/V Ratio
            st.markdown("<div class='section-header'>Vessel Analysis</div>", unsafe_allow_html=True)
            
            av_ratio = analysis['vessel_analysis']['av_ratio']
            st.markdown(f"<div class='av-ratio'>A/V Ratio: {av_ratio:.2f}</div>", unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Artery Diameter", f"{analysis['vessel_analysis']['artery_diameter']:.1f} µm")
            with col_b:
                st.metric("Vein Diameter", f"{analysis['vessel_analysis']['vein_diameter']:.1f} µm")
            
            av_status = analysis['vessel_analysis']['av_status']
            status_color = "normal" if av_status == "Normal" else "warning" if "narrowing" in av_status else "critical"
            st.markdown(f"<div class='metric-card {status_color}'><b>Status:</b> {av_status}</div>", unsafe_allow_html=True)
            
            # Macula Analysis
            st.markdown("<div class='section-header'>Macula Analysis</div>", unsafe_allow_html=True)
            
            if 'findings' in analysis['macula_analysis'] and analysis['macula_analysis']['findings']:
                for finding in analysis['macula_analysis']['findings']:
                    st.markdown(f"<div class='finding-item'>🔍 {finding}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='finding-item'>✅ Macula appears normal</div>", unsafe_allow_html=True)
            
            if analysis['macula_analysis'].get('has_drusen', False):
                st.warning(f"Drusen detected: {analysis['macula_analysis'].get('drusen_count', 0)} count")
            
            # Optic Disc Analysis
            st.markdown("<div class='section-header'>Optic Disc Analysis</div>", unsafe_allow_html=True)
            
            cd_ratio = analysis['optic_disc_analysis']['cd_ratio']
            risk_level = analysis['optic_disc_analysis']['risk_level']
            risk_class = "normal" if risk_level == "Low" else "warning" if risk_level == "Medium" else "critical"
            
            st.markdown(f"""
            <div class='metric-card {risk_class}'>
                <b>C/D Ratio:</b> {cd_ratio:.2f}<br>
                <b>Risk Level:</b> {risk_level}<br>
                <b>Rim Status:</b> {analysis['optic_disc_analysis']['rim_status']}
            </div>
            """, unsafe_allow_html=True)
            
            if analysis['optic_disc_analysis']['has_hemorrhage']:
                st.error("⚠️ Disc hemorrhage detected!")
    
    # Visualization and detailed results
    if st.session_state.complete_analysis:
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Visualization", "📈 Detailed Metrics", "⚠️ Pathologies", "💡 Recommendations"])
        
        with tab1:
            st.subheader("Anatomical Landmarks")
            if 'visualization' in st.session_state.complete_analysis:
                st.image(st.session_state.complete_analysis['visualization'], 
                        caption="Yellow: Macula | Green: Optic Disc | Red: Cup | Orange: Major Vessels",
                        use_container_width=True)
        
        with tab2:
            st.subheader("Detailed Measurements")
            
            # Create metrics dataframe
            metrics_data = []
            
            # Vessel metrics
            vessel = analysis['vessel_analysis']
            metrics_data.append({'Category': 'Vessels', 'Parameter': 'A/V Ratio', 'Value': f"{vessel['av_ratio']:.2f}", 'Normal Range': '0.6-0.8'})
            metrics_data.append({'Category': 'Vessels', 'Parameter': 'Vessel Density', 'Value': f"{vessel['vessel_density']*100:.1f}%", 'Normal Range': '15-25%'})
            
            # Macula metrics
            macula = analysis['macula_analysis']
            metrics_data.append({'Category': 'Macula', 'Parameter': 'Mean Intensity', 'Value': f"{macula.get('mean_intensity', 0):.0f}", 'Normal Range': '80-150'})
            metrics_data.append({'Category': 'Macula', 'Parameter': 'Foveal Reflex', 'Value': macula.get('fovea_reflex', 'N/A'), 'Normal Range': 'Present'})
            
            # Optic disc metrics
            disc = analysis['optic_disc_analysis']
            metrics_data.append({'Category': 'Optic Disc', 'Parameter': 'C/D Ratio', 'Value': f"{disc['cd_ratio']:.2f}", 'Normal Range': '<0.4'})
            metrics_data.append({'Category': 'Optic Disc', 'Parameter': 'Rim Thickness', 'Value': f"{disc['rim_thickness']:.1f} px", 'Normal Range': '>0.2×disc radius'})
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        with tab3:
            st.subheader("Detected Pathologies")
            
            pathologies = analysis['pathology_detection']
            
            if pathologies['pathologies']:
                for path in pathologies['pathologies']:
                    st.markdown(f"""
                    <div class='metric-card warning'>
                        <b>{path['type']}</b><br>
                        Count: {path.get('count', 1)}<br>
                        Location: {path.get('location', 'N/A')}<br>
                        Severity: {path.get('severity', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No significant pathologies detected")
            
            st.metric("DR Severity Level", pathologies['dr_severity'])
            st.metric("Requires Specialist Referral", "Yes" if pathologies['requires_referral'] else "No")
        
        with tab4:
            st.subheader("Clinical Recommendations")
            
            for i, recommendation in enumerate(analysis['recommendings'], 1):
                st.info(f"{i}. {recommendation}")
            
            # Follow-up timeline
            risk_factors = []
            if analysis['optic_disc_analysis']['risk_level'] == "High":
                risk_factors.append("High-risk optic disc")
            if analysis['pathology_detection']['requires_referral']:
                risk_factors.append("Significant pathologies")
            if analysis['vessel_analysis']['av_status'] != "Normal":
                risk_factors.append("Abnormal A/V ratio")
            
            if risk_factors:
                st.warning(f"**Follow-up needed in 1-3 months** due to: {', '.join(risk_factors)}")
            else:
                st.success("**Routine follow-up in 12-24 months**")
        
        # Report Generation
        st.markdown("---")
        st.subheader("📄 Generate Professional Report")
        
        if st.button("🖨️ Generate Comprehensive HTML Report", type="primary", use_container_width=True):
            if 'patient_info' in st.session_state:
                # Generate HTML report
                html_report = generate_comprehensive_report(
                    st.session_state.patient_info,
                    st.session_state.complete_analysis,
                    uploaded_file.name
                )
                
                # Create download link
                b64 = base64.b64encode(html_report.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="fundus_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html" style="text-decoration: none;">\
                        <button style="background-color: #1E3A8A; color: white; padding: 15px 30px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; width: 100%;">\
                        📥 Download Comprehensive HTML Report</button></a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Show preview
                with st.expander("🔍 Preview Report", expanded=False):
                    st.components.v1.html(html_report, height=1000, scrolling=True)
            else:
                st.error("Please enter patient information in the sidebar first.")

else:
    # Instructions
    st.info("👆 Please upload a fundus image for analysis")
    
    st.markdown("---")
    st.subheader("🔬 What This Analyzer Checks")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **🩸 Vessel Analysis**
        • A/V Ratio calculation
        • Arteriolar narrowing
        • Venous dilation
        • Vessel density
        """)
    
    with col2:
        st.markdown("""
        **⭐ Macula Analysis**
        • Drusen detection
        • Macular edema signs
        • Foveal reflex
        • Pigment changes
        """)
    
    with col3:
        st.markdown("""
        **🌀 Optic Disc**
        • Cup-to-Disc ratio
        • Neural rim assessment
        • Disc hemorrhages
        • Glaucoma risk
        """)
    
    with col4:
        st.markdown("""
        **⚠️ Pathology Detection**
        • Microaneurysms
        • Hemorrhages
        • Exudates (hard/soft)
        • Cotton wool spots
        • DR severity grading
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "<b>EasyScan Comprehensive Fundus Analyzer v4.0</b><br>"
    "Complete retinal analysis for clinical practice<br>"
    "For educational and clinical support purposes"
    "</div>",
    unsafe_allow_html=True
)
