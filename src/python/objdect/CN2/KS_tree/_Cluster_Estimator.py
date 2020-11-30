# -*- coding: utf-8 -*-
"""
"""

# TODO: If more than one colour channel present, check if cross channel values
# overlap; if so return one cluster (assumption that one entity present), 
# otherwise return 2/3 clusters (assumption that more than one entities present)
# Need to verify above claim.

def ClusterEstimator():
# Initialize lists to output results
    out_binlocs_l = []
    out_win_vals_l = []
    
    # Initialize number of clusters
    n_clusters = 2    # Starting from 2 since background is a cluster & possible
                      # artifacts
    
    # Enumerate histogram bins
    bin_c = np.arange(0, 256, dtype=np.uint8)
    
    for i in range(8):
        # If all channels present
        if np.logical_and.reduce((np.any(final_win_loc[i, :, 0]) != 0,
                                  np.any(final_win_loc[i, :, 1]) != 0,
                                  np.any(final_win_loc[i, :, 2]) != 0)):
            # Initialize lists to store values
            out_binlocs_list = []
            out_win_vals_list = []
            # Append bin locations
            out_binlocs_list.append(final_win_loc[i, :, :])
            
            # Append windows values
            out_win_vals_list.append(final_win_vals[i, :, :])
            
            # Convert resulting lists to pandas Series objects
            binlocs_pd = pd.Series(out_binlocs_list)
            win_vals_pd = pd.Series(out_win_vals_list)
            
            # Clear lists
            out_binlocs_list.clear()
            out_win_vals_list.clear()
            
            # Store results to arrays
            out_binlocs_temp = binlocs_pd[0]
            out_win_vals_temp = win_vals_pd[0]
            
            # Check if current windows have the same bin location for each
            # Colour channel
            if np.logical_and(out_binlocs_temp[:, 0].all() == out_binlocs_temp[:, 1].all(),
                              out_binlocs_temp[:, 1].all() == out_binlocs_temp[:, 2].all()):
                
                out_binlocs_l.append(out_binlocs_temp)
                out_win_vals_l.append(out_win_vals_temp)
                
                # Increment number of clusters by one
                n_clusters += 1
                
            # If the bin location is not the same for any colour channel
            else:
                # Perform KS-test
                out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                
                out_binlocs_l.append(out_binlocs)
                out_win_vals_l.append(out_win_vals)
                
                # Check if bin locations remain the same
                if np.logical_and(out_binlocs[:, 0].all() == out_binlocs[:, 1].all(),
                                  out_binlocs[:, 1].all() == out_binlocs[:, 2].all()):
                    # Increment number of clusters by 1
                    n_clusters += 1
                # Check if Blue & Green bin locations same, but not Red's
                elif np.logical_and(out_binlocs[:, 0].all() == out_binlocs[:, 1].all(),
                                    out_binlocs[:, 1].all() != out_binlocs[:, 2].all()):
                    # Increment number of clusters by 2
                    n_clusters += 2
                    
                # Check if Blue & Red bin locations same, but not Green's
                elif np.logical_and(out_binlocs[:, 0].all() == out_binlocs[:, 2].all(),
                                    out_binlocs[:, 2].all() != out_binlocs[:, 1].all()):
                    # Increment number of cluster by 2
                    n_clusters += 2
                    
                # Check if Red & Green bin locations same, but not Blue's
                elif np.logical_and(out_binlocs[:, 1].all() == out_binlocs[:, 2].all(),
                                    out_binlocs[:, 2].all() != out_binlocs[:, 0].all()):
                    # Increment number of clusters by 2
                    n_clusters += 2
                    
                # If every colour's bin locations unequal
                else:
                    # Increment number of clusters by 3
                    n_clusters += 3
                    
        # If Blue channel present
        elif np.any(final_win_loc[i, :, 0]) != 0:
        # If Blue and Red channels present
            if np.any(final_win_loc[i, :, 2]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 0] == out_binlocs_temp[:, 2]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 0].all() == out_binlocs[:, 2].all():
                        # Increment number of clusters by 1
                        n_clusters += 1
                    # If Blue & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            # If Blue & Green channes present
            elif np.any(final_win_loc[i, :, 1]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 0] == out_binlocs_temp[:, 1]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 0].all() == out_binlocs[:, 1].all():
                        # Increment number of clusters by 1
                        n_clusters += 1
                    # If Blue & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            
            # If only Blue channel present
            else:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                out_win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = out_win_vals_pd[0]
                
                out_binlocs_l.append(out_binlocs_temp)
                out_win_vals_l.append(out_win_vals_temp)
                
                # Increment number of clusters by 1
                n_clusters += 1
        # If Green channel present
        elif np.any(final_win_loc[i, :, 1]) != 0:
            # If Green & Blue channels present
            if np.any(final_win_loc[i, :, 0]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                out_win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = out_win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 0] == out_binlocs_temp[:, 1]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 0].all() == out_binlocs[:, 1].all():
                        # Increment number of clusters by 1
                        n_clusters += 1
                    # If Blue & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            # If Green & Red channes present
            elif np.any(final_win_loc[i, :, 2]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 1] == out_binlocs_temp[:, 2]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 1].all() == out_binlocs[:, 2].all():
                        
                        # Increment number of clusters by 1
                        n_clusters += 1
                    
                    # If Green & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            
            # If only Green channel present
            else:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                out_binlocs_l.append(out_binlocs_temp)
                out_win_vals_l.append(out_win_vals_temp)
                
                # Increment number of clusters by 1
                n_clusters += 1
        # If Red channel present
        elif np.any(final_win_loc[i, :, 2]) != 0:
            # If Red & Blue present
            if np.any(final_win_loc[i, :, 0]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 0] == out_binlocs_temp[:, 2]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 0].all() == out_binlocs[:, 2].all():
                        # Increment number of clusters by 1
                        n_clusters += 1
                    # If Blue & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            # If Red & Green channes present
            elif np.any(final_win_loc[i, :, 1]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 2] == out_binlocs_temp[:, 1]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 2].all() == out_binlocs[:, 1].all():
                        # Increment number of clusters by 1
                        n_clusters += 1
                    # If Blue & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            
            # If only Red channel present
            else:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                out_binlocs_l.append(out_binlocs_temp)
                out_win_vals_l.append(out_win_vals_temp)
                
                # Increment number of clusters by 1
                n_clusters += 1
        # If no colour channel present
        else:
            # Continue to next windows
            continue
    
    # Convert output lists into pandas Series objects
    out_binlocs_pd = pd.Series(out_binlocs_l)
    
    out_win_vals_pd = pd.Series(out_win_vals_l)
    
    # Test if any results are present
    try:
        out_binlocs_pd[0]
    except:
        # If not present, set outputs to zero
        out_binlocs = 0
        out_win_vals = 0
    else:
        # Initialize output arrays
        out_binlocs = np.zeros((np.int8(len(out_binlocs_pd)),
                                np.int8(len(out_binlocs_pd[0])), 3), dtype=np.uint8)
        
        out_win_vals = np.zeros((np.int8(len(out_binlocs_pd)),
                                 np.int8(len(out_binlocs_pd[0])), 3), dtype=np.float32)
        
        # Loop & store each window
        for i in range(len(out_binlocs_pd)):
            out_binlocs[i, :, :] = out_binlocs_pd[i]
            
            out_win_vals[i, :, :] = out_win_vals_pd[i]
    
    # If flag is set to 1, output the resulting windows
    if deb_flg == 1:
        out = pd.Series((out_binlocs, out_win_vals, n_clusters))
        return(out)
    # Else, return the number of estimated clusters
    else:
        return(n_clusters)