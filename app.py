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

# App title only
st.title("LiDAR Pothole Detection System")

# Create tabs immediately below the title
tab1, tab2, tab3, tab4 = st.tabs(["Original Cloud", "Pothole Detection", "Map View", "Export"])

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
def process_las_file(las_file, sample_size=None):
    with st.spinner("Processing LAS file..."):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmp_file:
            tmp_file.write(las_file.getvalue())
            tmp_path = tmp_file.name
        
        # Read the LAS file
        las = laspy.read(tmp_path)
        
        # Get point count and sample if necessary
        point_count = len(las.points)
        st.sidebar.write(f"Total points in LAS file: {point_count:,}")
        
        if sample_size and point_count > sample_size:
            indices = np.random.choice(point_count, sample_size, replace=False)
            st.sidebar.write(f"Sampling {sample_size:,} points for processing")
        else:
            indices = np.arange(point_count)
        
        # Extract coordinates
        x = np.array(las.x[indices], dtype=np.float64)
        y = np.array(las.y[indices], dtype=np.float64)
        z = np.array(las.z[indices], dtype=np.float64)
        
        # Extract color information if available
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            r = np.array(las.red[indices], dtype=np.float64) / 65535.0
            g = np.array(las.green[indices], dtype=np.float64) / 65535.0
            b = np.array(las.blue[indices], dtype=np.float64) / 65535.0
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
            
        return data

# Function to run inference with the Random Forest model
def run_inference(data, model, scaler):
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
        
        return data

# Function to visualize the original point cloud in RGB
def visualize_original_point_cloud(data, point_size, max_points=100000):
    with st.spinner("Preparing original point cloud visualization..."):
        # Sample data if too large
        if len(data) > max_points:
            st.info(f"Sampling {max_points:,} points for visualization")
            sample_data = data.sample(max_points)
        else:
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
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Function to visualize the point cloud with pothole predictions
def visualize_pothole_detection(data, point_size, max_points=100000):
    with st.spinner("Preparing pothole detection visualization..."):
        # Sample data if too large
        if len(data) > max_points:
            st.info(f"Sampling {max_points:,} points for visualization")
            sample_data = data.sample(max_points)
        else:
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
            margin=dict(l=0, r=0, b=0, t=0)
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
                volume = hull.volume
                
                # Calculate centroid
                centroid = cluster_points[['x', 'y', 'z']].mean().values
                
                volumes.append({
                    'cluster_id': i,
                    'volume': volume,
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
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=18)
                
                # Add pothole markers with popup showing volume information
                for _, row in volumes_df.iterrows():
                    # Scale marker size based on volume
                    radius = min(20, max(5, np.log(row['volume'] + 1) * 2))
                    
                    popup_html = f"""
                    <div style="font-family: Arial; width: 200px;">
                        <h4>Pothole Cluster {row['cluster_id']}</h4>
                        <b>Volume:</b> {row['volume']:.2f} cubic units<br>
                        <b>Point Count:</b> {row['point_count']}<br>
                    </div>
                    """
                    
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=radius,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(m)
                
                # Display the map with explicit dimensions
                folium_static(m, width=800, height=600)
                
            except Exception as e:
                st.error(f"Error creating map with CRS transformation: {e}")
                st.info("Displaying pothole locations in original coordinate system instead")
                display_local_map(volumes_df)
        else:
            # If no CRS, display in local coordinates
            display_local_map(volumes_df)

# Function to display pothole locations in local coordinates
def display_local_map(volumes_df):
    # Create a simple scatter plot of pothole locations
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scale marker size based on volume
    sizes = np.log(volumes_df['volume'] + 1) * 10
    
    scatter = ax.scatter(
        volumes_df['centroid_x'],
        volumes_df['centroid_y'],
        c='red',
        s=sizes,
        alpha=0.7
    )
    
    # Add annotations for larger potholes
    for _, row in volumes_df.nlargest(5, 'volume').iterrows():
        ax.annotate(
            f"Vol: {row['volume']:.1f}",
            (row['centroid_x'], row['centroid_y']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    ax.set_title('Pothole Locations (Local Coordinates)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    
    # Add colorbar legend
    plt.colorbar(scatter, label='Volume')
    
    st.pyplot(fig)

# Function to prepare export data
def prepare_export_data(data):
    # Create a copy of the data with only the required columns
    export_df = data[['x', 'y', 'z', 'r', 'g', 'b', 'pred']].copy()
    
    # Rename the prediction column to be more descriptive
    export_df = export_df.rename(columns={'pred': 'predicted_label'})
    
    return export_df

# Main application flow
def main():
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Processing options
    sample_size = st.sidebar.number_input(
        "Sample size for processing (0 for all points)",
        min_value=0,
        max_value=1000000,
        value=100000,
        step=10000
    )
    
    if sample_size == 0:
        sample_size = None
    
    # Add visualization settings
    st.sidebar.header("Visualization Settings")
    point_size = st.sidebar.slider(
        "Point Size", 
        min_value=1, 
        max_value=10, 
        value=4,
        step=1,
        help="Adjust the size of points in 3D visualizations"
    )
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload a .las file", type=["las"])
    
    # Process file if uploaded
    if uploaded_file and model is not None and scaler is not None:
        try:
            # Process the file
            data, las, crs, xyz = process_las_file(uploaded_file, sample_size)
            
            # Compute geometric features before inference
            data = compute_features_for_detection(data, xyz)
            
            # Run inference
            results = run_inference(data, model, scaler)
            
            # Calculate pothole volumes
            volumes_df = calculate_pothole_volumes(results)
            
            # Fill the tabs with content
            with tab1:
                # Original point cloud visualization
                st.subheader("Original Point Cloud (RGB)")
                visualize_original_point_cloud(data, point_size)
            
            with tab2:
                # Pothole detection visualization
                st.subheader("Pothole Detection Results")
                st.write("Red: Potholes | Blue: Non-road surfaces")
                visualize_pothole_detection(results, point_size)
            
            with tab3:
                # Map view
                st.subheader("Pothole Map")
                st.write("Click on markers to see pothole volume information")
                if volumes_df is not None and not volumes_df.empty:
                    display_pothole_map(results, volumes_df, crs)
                else:
                    st.info("No pothole clusters identified for mapping")
            
            with tab4:
                # Export options
                st.subheader("Export Data")
                
                # Prepare export data with xyz, rgb, and predicted labels
                export_df = prepare_export_data(results)
                
                # Display the data preview
                st.dataframe(export_df.head(100))  # Show first 100 rows
                
                # Option to download results
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download point cloud with predictions (CSV)",
                    data=csv,
                    file_name="lidar_pothole_detection.csv",
                    mime="text/csv",
                )
                
                # Show data statistics
                st.subheader("Data Statistics")
                total_points = len(export_df)
                pothole_points = export_df['predicted_label'].sum()
                
                st.write(f"Total points: {total_points:,}")
                st.write(f"Pothole points: {pothole_points:,} ({pothole_points/total_points*100:.2f}%)")
                st.write(f"Non-pothole points: {total_points-pothole_points:,} ({(total_points-pothole_points)/total_points*100:.2f}%)")
                
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.info("Please check your file and try again.")
    else:
        # Show placeholder content in tabs when no file is uploaded
        with tab1:
            st.info("Please upload a LAS file to view the original point cloud.")
        
        with tab2:
            st.info("Please upload a LAS file to detect potholes.")
        
        with tab3:
            st.info("Please upload a LAS file to view pothole locations on a map.")
        
        with tab4:
            st.info("Please upload a LAS file to export data.")

if __name__ == "__main__":
    main()
