import streamlit as st
import laspy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import os
import tempfile
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import time
import pyproj
from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

# Set page config (must be the first Streamlit command)
st.set_page_config(
    page_title="LiDAR Pothole Detection System",
    layout="wide"
)

# Add custom CSS for full-width map
st.markdown("""
<style>
.folium-map {
    width: 100% !important;
    height: 800px !important;
}
[data-testid="stMetricValue"] {
    font-size: 50px;
}
[data-testid="stMetricLabel"] {
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching processed data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'las' not in st.session_state:
    st.session_state.las = None
if 'crs' not in st.session_state:
    st.session_state.crs = None
if 'xyz' not in st.session_state:
    st.session_state.xyz = None
if 'features_computed' not in st.session_state:
    st.session_state.features_computed = False
if 'inference_done' not in st.session_state:
    st.session_state.inference_done = False
if 'last_file_name' not in st.session_state:
    st.session_state.last_file_name = None

# App title only
st.title("LiDAR Pothole Detection System")

# Create tabs immediately below the title
tab1, tab2, tab3, tab4 = st.tabs(["Original Cloud", "Pothole Detection", "Map View", "Export"])

# Sidebar controls
point_size = st.sidebar.slider("Point Size", 1, 10, 3)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('random_forest.joblib')
        scaler = joblib.load('scaler.joblib')
        st.sidebar.success("✅ Model and scaler loaded successfully!")
        return model, scaler
    except Exception as e:
        st.sidebar.error(f"Error loading model or scaler: {e}")
        st.sidebar.info("Please make sure 'random_forest.joblib' and 'scaler.joblib' files are in the same directory as this app.")
        return None, None

# Function to compute geometric features (from the notebook)
def compute_geometric_features(xyz, radius=0.5, max_nn=30):
    # Create progress bar for this operation
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Creating point cloud...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Update progress
    progress_bar.progress(20)
    status_text.text("Estimating normals...")
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    normals = np.asarray(pcd.normals)
    
    # Update progress
    progress_bar.progress(50)
    status_text.text("Computing curvature and roughness...")
    
    # Use Nearest Neighbors for curvature, roughness
    nbrs = NearestNeighbors(n_neighbors=max_nn).fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)
    
    # Initialize arrays for better performance
    total_points = len(indices)
    curvatures = np.zeros(total_points)
    roughnesses = np.zeros(total_points)
    
    # Process in batches to show progress
    batch_size = max(1, total_points // 10)
    for i in range(0, total_points, batch_size):
        end_idx = min(i + batch_size, total_points)
        batch_indices = indices[i:end_idx]
        for j, idx in enumerate(batch_indices):
            neighbors = xyz[idx]
            cov = np.cov(neighbors.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)
            curvatures[i+j] = eigvals[0] / eigvals.sum() if eigvals.sum() > 0 else 0
            roughnesses[i+j] = np.std(neighbors[:, 2])  # std of z
        
        # Update progress
        progress_percent = min(50 + int(50 * end_idx / total_points), 100)
        progress_bar.progress(progress_percent)
        status_text.text(f"Processing features: {end_idx}/{total_points} points")
    
    progress_bar.progress(100)
    status_text.text("Feature computation complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return normals, curvatures, roughnesses

# Function to process LAS file and prepare features
def process_las_file(las_file):
    with st.spinner("Processing LAS file..."):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmp_file:
            tmp_file.write(las_file.getvalue())
            tmp_path = tmp_file.name
        
        # Read the LAS file
        las = laspy.read(tmp_path)
        
        # Get point count
        point_count = len(las.points)
        st.sidebar.write(f"Total points in LAS file: {point_count:,}")
        
        # Extract coordinates - use all points, no sampling
        x = np.array(las.x, dtype=np.float64)
        y = np.array(las.y, dtype=np.float64)
        z = np.array(las.z, dtype=np.float64)
        
        # Extract color information if available
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            r = np.array(las.red, dtype=np.float64) / 65535.0
            g = np.array(las.green, dtype=np.float64) / 65535.0
            b = np.array(las.blue, dtype=np.float64) / 65535.0
        else:
            st.sidebar.warning("No color data found in LAS file. Using default values.")
            r = np.zeros_like(x)
            g = np.zeros_like(x)
            b = np.zeros_like(x)
        
        # Get coordinate reference system if available
        crs = None
        if hasattr(las.header, 'parse_crs'):
            try:
                crs = las.header.parse_crs()
                st.sidebar.info(f"CRS detected: {crs}")
            except:
                st.sidebar.warning("Could not parse CRS from LAS file")
        
        # Stack coordinates
        xyz = np.column_stack((x, y, z))
        
        # Create DataFrame with basic info
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'r': r,
            'g': g,
            'b': b,
        })
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return df, las, crs, xyz

# Function to compute geometric features when needed
def compute_features_for_detection(data, xyz):
    # Check if features have already been computed
    if st.session_state.features_computed:
        return st.session_state.data
    
    with st.spinner("Computing geometric features for detection..."):
        # Compute geometric features with progress tracking
        st.sidebar.write("Computing geometric features...")
        normals, curvature, roughness = compute_geometric_features(xyz)
        
        # Add features to DataFrame
        data['nx'] = normals[:, 0]
        data['ny'] = normals[:, 1]
        data['nz'] = normals[:, 2]
        data['curvature'] = curvature
        data['roughness'] = roughness
        
        # Fill any missing values
        if data.isnull().any().any():
            st.sidebar.warning("Filling NaN values with column means...")
            data = data.fillna(data.mean())
        
        # Update session state
        st.session_state.data = data
        st.session_state.features_computed = True
        
        return data

# Function to run inference with the Random Forest model
def run_inference(data, model, scaler):
    # Check if inference has already been done
    if st.session_state.inference_done:
        return st.session_state.data
    
    with st.spinner("Running pothole detection..."):
        features = ['x', 'y', 'z', 'r', 'g', 'b', 'nx', 'ny', 'nz', 'curvature', 'roughness']
        
        # Check if all required features exist
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
            st.stop()
        
        # Prepare features for inference
        X = data[features].to_numpy(dtype=np.float64)
        
        # Fill any NaN values
        if np.isnan(X).any():
            st.sidebar.warning("Replacing NaN values...")
            column_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X)
            for i in range(X.shape[1]):
                X[nan_mask[:, i], i] = column_means[i]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Handle any remaining NaN/inf values
        X_scaled = np.nan_to_num(X_scaled)
        
        # Make predictions
        preds = model.predict(X_scaled)
        pothole_count = np.sum(preds == 1)
        st.sidebar.success(f"Found {pothole_count:,} potential potholes out of {len(preds):,} points ({pothole_count/len(preds)*100:.2f}%)")
        
        # Add predictions to original data
        data['pred'] = preds
        
        # Update session state
        st.session_state.data = data
        st.session_state.inference_done = True
        
        return data

# Function to visualize the original point cloud in RGB
def visualize_original_point_cloud(data, point_size):
    with st.spinner("Preparing original point cloud visualization..."):
        # Use all data points, no sampling
        sample_data = data
        
        # Prepare data for visualization
        xyz = sample_data[['x', 'y', 'z']].values
        
        # Normalize for better visualization
        xyz_norm = xyz.copy()
        xyz_norm -= xyz_norm.min(axis=0)
        xyz_norm /= xyz_norm.max(axis=0) if xyz_norm.max(axis=0).max() > 0 else 1
        
        # Flatten z-axis to make it appear more flat
        xyz_norm[:, 2] = xyz_norm[:, 2] * 0.3  # Moderate depth scaling
        
        # Use RGB colors from the point cloud
        colors = sample_data[['r', 'g', 'b']].values
        
        # Create Plotly figure
        fig = go.Figure(data=[go.Scatter3d(
            x=xyz_norm[:, 0],
            y=xyz_norm[:, 1],
            z=xyz_norm[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,  # Use the slider value
                color=[f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors],
                opacity=0.8
            ),
            hoverinfo='none'
        )])
        
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(showgrid=False, zeroline=False),  # Remove grid
                yaxis=dict(showgrid=False, zeroline=False),  # Remove grid
                zaxis=dict(showgrid=False, zeroline=False)   # Remove grid
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=800,  # Increased height
            width=1200   # Increased width
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Function to visualize the point cloud with pothole predictions
def visualize_pothole_detection(data, point_size):
    with st.spinner("Preparing pothole detection visualization..."):
        # Use all data points, no sampling
        sample_data = data
        
        # Prepare data for visualization
        xyz = sample_data[['x', 'y', 'z']].values
        
        # Normalize for better visualization
        xyz_norm = xyz.copy()
        xyz_norm -= xyz_norm.min(axis=0)
        xyz_norm /= xyz_norm.max(axis=0) if xyz_norm.max(axis=0).max() > 0 else 1
        
        # Flatten z-axis to make it appear more flat
        xyz_norm[:, 2] = xyz_norm[:, 2] * 0.3  # Moderate depth scaling
        
        # Color points based on prediction
        colors = np.zeros((len(sample_data), 3))
        pothole_mask = sample_data['pred'] == 1
        non_pothole_mask = ~pothole_mask
        
        # Set colors: red for potholes, blue for non-road surfaces
        colors[pothole_mask] = [1, 0, 0]  # Red for potholes
        colors[non_pothole_mask] = [0, 0, 1]  # Blue for non-road surfaces
        
        # Create Plotly figure
        fig = go.Figure(data=[go.Scatter3d(
            x=xyz_norm[:, 0],
            y=xyz_norm[:, 1],
            z=xyz_norm[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,  # Use the slider value
                color=[f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors],
                opacity=0.8
            ),
            hoverinfo='none'
        )])
        
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(showgrid=False, zeroline=False),  # Remove grid
                yaxis=dict(showgrid=False, zeroline=False),  # Remove grid
                zaxis=dict(showgrid=False, zeroline=False)   # Remove grid
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=800,  # Increased height
            width=1200   # Increased width
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Function to calculate pothole volumes
def calculate_pothole_volumes(data, min_points=10):
    with st.spinner("Calculating pothole volumes..."):
        # Get only pothole points
        pothole_data = data[data['pred'] == 1]
        
        if len(pothole_data) == 0:
            st.warning("No potholes detected")
            return None
        
        # Extract coordinates
        xyz = pothole_data[['x', 'y', 'z']].values
        
        # Determine eps parameter based on data scale
        x_range = xyz[:, 0].max() - xyz[:, 0].min()
        y_range = xyz[:, 1].max() - xyz[:, 1].min()
        avg_range = (x_range + y_range) / 2
        eps = avg_range / 100  # Adjust as needed
        
        # Run DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_points).fit(xyz)
        labels = db.labels_
        
        # Add cluster labels to the data
        pothole_data['cluster'] = labels
        
        # Count clusters (excluding noise with label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        st.sidebar.write(f"Identified {n_clusters} distinct pothole clusters")
        
        # Calculate volume for each cluster
        volumes = []
        for i in range(n_clusters):
            cluster_points = pothole_data[pothole_data['cluster'] == i]
            if len(cluster_points) < 4:  # Need at least 4 points for ConvexHull
                continue
            
            try:
                # Use ConvexHull to calculate volume
                hull = ConvexHull(cluster_points[['x', 'y', 'z']].values)
                volume_m3 = hull.volume
                volume_cm3 = volume_m3 * 1000000  # Convert to cubic centimeters
                
                # Calculate centroid
                centroid = cluster_points[['x', 'y', 'z']].mean().values
                
                volumes.append({
                    'cluster_id': i,
                    'volume_m3': volume_m3,
                    'volume_cm3': volume_cm3,
                    'point_count': len(cluster_points),
                    'centroid_x': centroid[0],
                    'centroid_y': centroid[1],
                    'centroid_z': centroid[2]
                })
            except Exception as e:
                st.sidebar.warning(f"Could not calculate volume for cluster {i}: {e}")
        
        # Create DataFrame with volumes
        if volumes:
            volumes_df = pd.DataFrame(volumes)
            return volumes_df
        else:
            st.sidebar.warning("Could not calculate volumes for any pothole clusters")
            return None

# Function to display pothole locations on a map with clickable markers
def display_pothole_map(data, volumes_df, crs=None):
    with st.spinner("Creating pothole map..."):
        if volumes_df is None or len(volumes_df) == 0:
            st.warning("No pothole data available for mapping")
            return
            
        # Create a map centered at the mean of the data
        center_x = volumes_df['centroid_x'].mean()
        center_y = volumes_df['centroid_y'].mean()
        
        # If we have a CRS, convert to lat/lon
        if crs:
            try:
                # Create transformer to convert to WGS84 (lat/lon)
                transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                
                # Transform centroids to lat/lon
                lons, lats = transformer.transform(
                    volumes_df['centroid_x'].values,
                    volumes_df['centroid_y'].values
                )
                volumes_df['lon'] = lons
                volumes_df['lat'] = lats
                
                center_lon, center_lat = transformer.transform(center_x, center_y)
                
                # Create map with larger dimensions
                m = folium.Map(location=[center_lat, center_lon], zoom_start=18, width='100%', height='100%')
                
                # Add pothole markers with popup showing volume information
                for _, row in volumes_df.iterrows():
                    # Determine icon color based on volume size
                    if row['volume_cm3'] > 10000:
                        icon_color = 'red'
                        severity = 'High'
                    elif row['volume_cm3'] > 5000:
                        icon_color = 'orange'
                        severity = 'Medium'
                    else:
                        icon_color = 'blue'
                        severity = 'Low'
                    
                    popup_html = f"""
                    <div style="width: 250px">
                        <h4 style="color:{icon_color}; text-align:center">Pothole #{int(row['cluster_id'])}</h4>
                        <p><b>Volume:</b> {row['volume_cm3']:.2f} cm³</p>
                        <p><b>Point Count:</b> {row['point_count']}</p>
                        <p><b>Location:</b> ({row['lat']:.6f}, {row['lon']:.6f})</p>
                        <p><b>Severity:</b> {severity}</p>
                    </div>
                    """
                    
                    # Use a more visible icon for potholes
                    folium.Marker(
                        location=[row['lat'], row['lon']],
                        popup=folium.Popup(popup_html, max_width=300),
                        icon=folium.Icon(color=icon_color, icon='road', prefix='fa')
                    ).add_to(m)
                    
                    # Also add a circle to represent the size
                    folium.Circle(
                        location=[row['lat'], row['lon']],
                        radius=max(1, min(20, row['volume_cm3'] / 1000)),  # Scale radius based on volume
                        color=icon_color,
                        fill=True,
                        fill_color=icon_color,
                        fill_opacity=0.4
                    ).add_to(m)
                
                # Use folium_static with full width
                folium_static(m, width=1400, height=800)
            except Exception as e:
                st.error(f"Error creating map: {e}")
                st.info("If your LAS file doesn't have a valid CRS, mapping may not be possible.")
        else:
            st.warning("No coordinate reference system (CRS) found in the LAS file. Cannot create map.")

# Function to export pothole data
def export_pothole_data(data, volumes_df):
    with st.spinner("Preparing data for export..."):
        if 'pred' not in data.columns:
            st.warning("No pothole detection results available for export")
            return
        
        # Create columns for export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Point Cloud")
            
            # Export options for point cloud
            export_format = st.selectbox(
                "Select export format",
                ["CSV", "GeoJSON (if CRS available)"]
            )
            
            export_points = st.radio(
                "Points to export",
                ["All points", "Pothole points only"]
            )
            
            if st.button("Export Point Cloud"):
                # Filter data if needed
                if export_points == "Pothole points only":
                    export_data = data[data['pred'] == 1].copy()
                else:
                    export_data = data.copy()
                
                if len(export_data) == 0:
                    st.warning("No points to export")
                    return
                
                # Export based on selected format
                if export_format == "CSV":
                    csv = export_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="pothole_detection.csv",
                        mime="text/csv"
                    )
                else:  # GeoJSON
                    if st.session_state.crs:
                        try:
                            # Create GeoDataFrame
                            geometry = [Point(x, y) for x, y in zip(export_data['x'], export_data['y'])]
                            gdf = gpd.GeoDataFrame(export_data, geometry=geometry, crs=st.session_state.crs)
                            
                            # Convert to GeoJSON
                            geojson = gdf.to_json()
                            st.download_button(
                                label="Download GeoJSON",
                                data=geojson,
                                file_name="pothole_detection.geojson",
                                mime="application/json"
                            )
                        except Exception as e:
                            st.error(f"Error creating GeoJSON: {e}")
                    else:
                        st.warning("No CRS available. Cannot export as GeoJSON.")
        
        with col2:
            st.subheader("Export Pothole Analysis")
            
            if volumes_df is not None and len(volumes_df) > 0:
                # Export options for pothole analysis
                if st.button("Export Pothole Analysis"):
                    csv = volumes_df.to_csv(index=False)
                    st.download_button(
                        label="Download Pothole Analysis CSV",
                        data=csv,
                        file_name="pothole_analysis.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No pothole analysis available for export")

# Main application flow
def main():
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # File uploader in sidebar
    las_file = st.sidebar.file_uploader("Upload LAS file", type=['las', 'laz'])
    
    if las_file is not None:
        # Check if we need to process the file (new file or first run)
        file_changed = (st.session_state.last_file_name != las_file.name)
        
        if file_changed or st.session_state.data is None:
            # Reset session state for new file
            st.session_state.features_computed = False
            st.session_state.inference_done = False
            st.session_state.last_file_name = las_file.name
            
            # Process the LAS file
            df, las, crs, xyz = process_las_file(las_file)
            
            # Store in session state
            st.session_state.data = df
            st.session_state.las = las
            st.session_state.crs = crs
            st.session_state.xyz = xyz
        
        # Use data from session state
        data = st.session_state.data
        xyz = st.session_state.xyz
        crs = st.session_state.crs
        
        # Original Point Cloud tab
        with tab1:
            st.header("Original Point Cloud")
            visualize_original_point_cloud(data, point_size)
        
        # Pothole Detection tab
        with tab2:
            st.header("Pothole Detection")
            
            if model is not None and scaler is not None:
                # Compute features if not already done
                data = compute_features_for_detection(data, xyz)
                
                # Run inference if not already done
                data = run_inference(data, model, scaler)
                
                # Display pothole detection visualization
                visualize_pothole_detection(data, point_size)
                
                # Calculate pothole volumes
                volumes_df = calculate_pothole_volumes(data)
                
                # Display pothole statistics with cubic centimeters
                if volumes_df is not None and len(volumes_df) > 0:
                    st.subheader("Pothole Statistics")
                    
                    # Create metrics for total volume
                    total_volume_cm3 = volumes_df['volume_cm3'].sum()
                    avg_volume_cm3 = volumes_df['volume_cm3'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Potholes", f"{len(volumes_df)}")
                    with col2:
                        st.metric("Total Volume", f"{total_volume_cm3:.2f} cm³")
                    with col3:
                        st.metric("Average Volume", f"{avg_volume_cm3:.2f} cm³")
                    
                    # Show detailed table
                    st.dataframe(volumes_df[['cluster_id', 'volume_cm3', 'point_count', 'centroid_x', 'centroid_y', 'centroid_z']])
            else:
                st.warning("Model or scaler not loaded. Cannot perform pothole detection.")
        
        # Map View tab
        with tab3:
            st.header("Pothole Map View")
            
            if 'pred' in data.columns:
                volumes_df = calculate_pothole_volumes(data) if 'volumes_df' not in locals() else volumes_df
                display_pothole_map(data, volumes_df, crs)
            else:
                st.warning("Run pothole detection first to view results on map")
        
        # Export tab
        with tab4:
            st.header("Export Data")
            
            if 'pred' in data.columns:
                volumes_df = calculate_pothole_volumes(data) if 'volumes_df' not in locals() else volumes_df
                export_pothole_data(data, volumes_df)
            else:
                st.warning("Run pothole detection first to export results")
    else:
        # Display instructions when no file is uploaded
        st.info("Please upload a LAS file to begin pothole detection")

if __name__ == "__main__":
    main()
