from fpdf import FPDF
import math
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import nibabel as nib
import numpy as np
import os
import random
import sklearn.metrics as skl
import sys
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec
from skimage.measure import marching_cubes

def asofi(subjid, image, output_folder):
    subjid_dir = os.path.join(output_folder, subjid)
    os.makedirs(subjid_dir, exist_ok=True)
    
    out_dir = subjid_dir

    print('Started: image with prefix name %s', subjid)

    if isinstance(image, nib.Nifti1Image):
        img = image
    else:
        img = nib.load(image)
        
    nii_header = img.header
    imageloaded = img.get_fdata()

    voxels_size = nii_header['pixdim'][1:4]
    print('The voxel size is %s x %s x %s mm^3', voxels_size[0], voxels_size[1], voxels_size[2])
    if voxels_size[0] != voxels_size[1] or voxels_size[0] != voxels_size[2] or voxels_size[1] != voxels_size[2]:
        sys.exit('The voxel is not isotropic! Exit.')

    L_min = voxels_size[0]
    print('The minimum size of the image is %s mm', L_min)
    Ly = imageloaded.shape[0]
    Lx = imageloaded.shape[1]
    Lz = imageloaded.shape[2]
    if Lx > Ly:
        L_Max = Lx
    else:
        L_Max = Ly
    if Lz > L_Max:
        L_Max = Lz
    print('The maximum size of the image is %s mm', L_Max)

    voxels = np.argwhere(imageloaded > 0)
    print('The non-zero voxels in the image are (the image volume) %s', voxels.shape[0])

    Ns = []
    Ns_std = []
    scales = []
    stop = math.ceil(math.log2(L_Max))
    for exp in range(stop + 1):
        scales.append(2 ** exp)
    scales = np.asarray(scales)
    random.seed(1)

    for scale in scales:
        print('Computing scale %s...', scale)
        Ns_offset = []
        for i in range(20):
            y0_rand = -random.randint(0, scale)
            yend_rand = Ly + 1 + scale
            x0_rand = -random.randint(0, scale)
            xend_rand = Lx + 1 + scale
            z0_rand = -random.randint(0, scale)
            zend_rand = Lz + 1 + scale

            H, edges = np.histogramdd(voxels, bins=(
                np.arange(y0_rand, yend_rand, scale),
                np.arange(x0_rand, xend_rand, scale),
                np.arange(z0_rand, zend_rand, scale)))

            count = np.sum(H > 0)
            Ns_offset.append(count)
            print('======= Offset %s: x0_rand = %s, y0_rand = %s, z0_rand = %s, count = %s ', i + 1, x0_rand,
                     y0_rand, z0_rand, count)

        Ns.append(np.mean(Ns_offset))
        Ns_std.append(np.std(Ns_offset))

    minWindowSize = 5
    scales_indices = []

    for step in range(scales.size, minWindowSize - 1, -1):
        for start_index in range(0, scales.size - step + 1):
            scales_indices.append((start_index, start_index + step - 1))
    scales_indices = np.asarray(scales_indices)

    k_ind = 1
    R2_adj = -1

    for k in range(scales_indices.shape[0]):
        start = scales_indices[k, 0]
        end = scales_indices[k, 1]

        coeffs = np.polyfit(np.log2(scales)[start:end + 1], np.log2(Ns)[start:end + 1], 1)
        n = end - start + 1
        y_true = np.log2(Ns)[start:end + 1]
        y_pred = np.polyval(coeffs, np.log2(scales)[start:end + 1])
        R2 = skl.r2_score(y_true, y_pred)
        R2_adj_tmp = 1 - (1 - R2) * ((n - 1) / (n - (k_ind + 1)))

        print(
            'In the interval [%s, %s] voxels, the FD is %s and the determination coefficient adjusted for the number of points is %s',
            scales[start], scales[end], -coeffs[0], R2_adj_tmp)

        R2_adj = round(R2_adj, 3)
        R2_adj_tmp = round(R2_adj_tmp, 3)

        if R2_adj_tmp > R2_adj:
            R2_adj = R2_adj_tmp
            FD = -coeffs[0]
            mfs = scales[start]
            Mfs = scales[end]
            fsw_index = k
            coeffs_selected = coeffs

        FD = round(FD, 4)

    mfs = mfs * L_min
    Mfs = Mfs * L_min
    print('The mfs automatically selected is %s', mfs)
    print('The Mfs automatically selected is %s', Mfs)
    print('The FD automatically selected is %s', FD)
    print('The R2_adj is %s', R2_adj)
    print("mfs automatically selected:", mfs)
    print("Mfs automatically selected:", Mfs)
    print("FD automatically selected:", FD)

    plt.figure()
    plt.errorbar(scales, Ns, yerr=Ns_std, fmt='o', ecolor='red', capsize=4, label='Mean Count ± Std Dev')
    plt.plot(scales, Ns, 'b-', alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('scale')
    plt.ylabel('count')
    plt.title('Box Count Stability across Random Offsets')
    plt.legend()
    variance_path = os.path.join(subjid_dir, f"{subjid}_BoxCountVariance.png")
    plt.savefig(variance_path)
    plt.close()

    plt.figure()
    plt.plot(scales, Ns, 'o', mfc='none')
    start, end = scales_indices[fsw_index]
    x_fit_log = np.log2(scales[start:end + 1])
    y_fit_log = np.polyval(coeffs_selected, x_fit_log)
    plt.plot(scales[start:end + 1], np.power(2, y_fit_log))
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('$\\epsilon$ (mm)')
    plt.ylabel('N (-)')
    fdplot_path = os.path.join(subjid_dir, f"{subjid}_FD_plot.png")
    plt.savefig(fdplot_path)
    plt.close()

    plt.figure()
    x_vals = np.log2(scales)
    y_vals = np.log2(Ns)
    slopes = np.diff(y_vals) / np.diff(x_vals)
    local_FDs = -slopes
    plot_scales = np.sqrt(scales[:-1] * scales[1:])
    plt.plot(plot_scales, local_FDs, 'o-', color='purple', label='Local FD')
    plt.axhline(FD, color='g', linestyle='--', label=f'Global FD ({FD})')
    plt.xscale('log', base=2)
    plt.xlabel('Box Size (mm)')
    plt.ylabel('Local Fractal Dimension')
    plt.title('Variation of FD across Box Sizes')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.savefig(output_folder + '/' + subjid + '/' + subjid + '_' + '_LocalFD.png')
    plt.clf()
    plt.close()

    x_vals = np.log2(scales)
    y_vals = np.log2(Ns)
    slopes = np.diff(y_vals) / np.diff(x_vals)
    local_FDs = -slopes
    local_FDs = np.append(local_FDs, np.nan)

    viz_scales = []
    for s in scales:
        s_mm = s * L_min
        if s_mm >= (mfs - 0.001) and s_mm <= (Mfs + 0.001):
            viz_scales.append(s)

    if not viz_scales:
        viz_scales = [scales[scales_indices[fsw_index, 0]], scales[scales_indices[fsw_index, 1]]]

    fig, axes = plt.subplots(nrows=len(viz_scales), ncols=1, figsize=(8, 5 * len(viz_scales)))
    if len(viz_scales) == 1:
        axes = [axes]

    z_slice = Lz // 2
    slice_data = imageloaded[:, :, z_slice]

    for idx, s in enumerate(viz_scales):
        ax = axes[idx]
        s_mm = s * L_min
        ax.imshow(slice_data, cmap='gray', origin='lower')
        for x in range(0, Lx, int(s)):
            ax.axvline(x, color='r', linewidth=0.5, alpha=0.5)
        for y in range(0, Ly, int(s)):
            ax.axhline(y, color='r', linewidth=0.5, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        scale_original_idx = np.where(scales == s)[0][0]
        box_count = Ns[scale_original_idx]
        count_str = f"{int(box_count):,}"
        l_fd = local_FDs[scale_original_idx]
        if np.isnan(l_fd):
            fd_str = "End (N/A)"
        else:
            fd_str = f"{l_fd:.4f}"
        details_text = (
            f"Scale Step: {int(s) * int(s)} voxels\n"
            f"Physical Size: {s_mm:.2f} mm (box edge length)\n"
            f"Box Count (N): {count_str}\n"
            f"Local FD: {fd_str}"
        )
        ax.text(1.05, 0.5, details_text, transform=ax.transAxes,
                verticalalignment='center', horizontalalignment='left',
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.suptitle(f"Fractal Window Visualization\nGlobal FD: {FD} (over {mfs:.2f}-{Mfs:.2f}mm)", y=1.00)
    gridviz_path = output_folder + '/' + subjid + '/' + subjid + '_' + 'GridViz.png'
    
    plt.savefig(gridviz_path, bbox_inches='tight')
    
    plt.close()

    print("making combo plot")
    x_vals = np.log2(scales)
    y_vals = np.log2(Ns)
    slopes = np.diff(y_vals) / np.diff(x_vals)
    local_FDs = -slopes
    local_FDs = np.append(local_FDs, np.nan)

    valid_indices = []
    for i, s in enumerate(scales):
        s_mm = s * L_min
        if s_mm >= (mfs - 0.001) and s_mm <= (Mfs + 0.001):
            valid_indices.append(i)

    if len(valid_indices) >= 4:
        selected_indices = np.linspace(valid_indices[0], valid_indices[-1], 4, dtype=int)
    elif len(valid_indices) > 0:
        selected_indices = valid_indices
    else:
        selected_indices = [0, len(scales) - 1]

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, len(selected_indices), height_ratios=[1, 0.8], hspace=0.1, wspace=0.05)

    ax_2d = fig.add_subplot(gs[1, :])
    x_all = scales
    y_all = np.array(Ns)

    ax_2d.plot(x_all, y_all, 'o--', color='lightgray', label='All Scales')

    win_start, win_end = scales_indices[fsw_index, 0], scales_indices[fsw_index, 1]
    x_win = x_all[win_start:win_end + 1]
    x_win_log = np.log2(x_win)
    y_fit_win_log = np.polyval(coeffs_selected, x_win_log)
    y_fit_win = np.power(2, y_fit_win_log)

    ax_2d.plot(x_win, y_fit_win, 'k-', linewidth=2, label=f'Global Fit (FD={FD})')

    x_sel = x_all[selected_indices]
    y_sel = y_all[selected_indices]
    ax_2d.plot(x_sel, y_sel, 'o', color='purple', markersize=10, markerfacecolor='none', markeredgewidth=2,
               label='Visualized Scales')

    ax_2d.legend()
    ax_2d.set_xscale('log', base=2)
    ax_2d.set_yscale('log', base=2)
    ax_2d.set_xlabel('scale (s)')
    ax_2d.set_ylabel('Count (N)')
    ax_2d.set_title('Fractal Dimension Analysis')

    verts, faces, _, _ = marching_cubes(imageloaded, level=0.5, step_size=3)
    verts = verts[:, [1, 0, 2]]

    for i, idx in enumerate(selected_indices):
        s = scales[idx]
        s_mm = s * L_min
        current_N = Ns[idx]
        current_LFD = local_FDs[idx]
        lfd_str = f"{current_LFD:.3f}" if not np.isnan(current_LFD) else "N/A"

        ax_3d = fig.add_subplot(gs[0, i], projection='3d')
        mesh = ax_3d.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                                  color='purple', alpha=0.8, edgecolor='none', shade=True)

        ax_3d.set_xticks(np.arange(0, Lx, s))
        ax_3d.set_yticks(np.arange(0, Ly, s))
        ax_3d.set_zticks(np.arange(0, Lz, s))
        ax_3d.set_axisbelow(False)
        ax_3d.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
        ax_3d.set_xticklabels([])
        ax_3d.set_yticklabels([])
        ax_3d.set_zticklabels([])
        ax_3d.view_init(elev=-30, azim=45)
        ax_3d.set_box_aspect((Lx, Ly, Lz))

        title_text = (
            f"s = {int(s)} voxels ({s_mm:.1f} mm)\n"
            f"Boxes (N) = {int(current_N):,}\n"
            f"Local FD = {lfd_str}"
        )
        ax_3d.set_title(title_text, fontsize=10, y=1.05)
        xyA = (x_sel[i], y_sel[i])
        xyB = (0.5, 0.0)
        con = ConnectionPatch(xyA=xyA, xyB=xyB,
                              coordsA="data", coordsB="axes fraction",
                              axesA=ax_2d, axesB=ax_3d,
                              arrowstyle="-|>", color="gray", lw=1.5, linestyle="--",
                              shrinkB=10)
        ax_2d.add_artist(con)

    plt.suptitle(f"Multi-scale Structural Analysis: {subjid}\n(Surface + Grid Visualization)", fontsize=16)

    combo_path = output_folder + '/' + subjid + '/' + subjid + '_' + '_3D_Combo.png'
    
    plt.savefig(combo_path, bbox_inches='tight')
    plt.close()

    txt_path = os.path.join(subjid_dir, f"{subjid}_FractalIndices.txt")
    with open(txt_path, 'w') as f:
        f.write(f"mfs (mm), {mfs}\n")
        f.write(f"Mfs (mm), {Mfs}\n")
        f.write(f"FD (-), {FD}\n")

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Fractal Analysis Report - Subject {subjid}", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Subjid: {subjid}", ln=1)
    pdf.cell(0, 10, f"mfs: {mfs:.2f} mm", ln=1)
    pdf.cell(0, 10, f"Mfs: {Mfs:.2f} mm", ln=1)
    pdf.cell(0, 10, f"FD: {FD:.4f}", ln=1)
    pdf.cell(0, 10, f"Adjusted R²: {R2_adj:.4f}", ln=1)
    pdf.ln(10)

    def add_plot(title, path, w=180):
        if os.path.exists(path):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, title, ln=1, align='C')
            pdf.image(path, x=(210 - w) / 2, w=w)
            pdf.ln(10)
        else:
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"{title}: Image not available", ln=1)

    add_plot("Main Fractal Dimension Plot", fdplot_path)
    pdf.add_page()
    add_plot("Box Counting Grid Visualization", gridviz_path)
    add_plot("Box Count Stability (Error Bars)", variance_path)
    pdf.add_page()
    add_plot("Combo Plot", combo_path)

    pdf.output(os.path.join(subjid_dir, f"{subjid}_FD_summary.pdf"))
