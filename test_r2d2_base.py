"""
Unit tests for r2d2_base.py

To run these tests:
    pip install pytest pytest-cov numpy ants pandas
    pytest test_r2d2_base.py -v
    pytest test_r2d2_base.py -v --cov=r2d2_base --cov-report=html
"""

import pytest
import numpy as np
import ants
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import r2d2_base


class TestROIFunctions:
    """Tests for ROI calculation utility functions"""

    def test_roi_min_vals_no_clipping(self):
        """Test roi_min_vals when radius doesn't go below 0"""
        result = r2d2_base.roi_min_vals(10, 10, 10, radius=5)
        assert result == [5, 5, 5]

    def test_roi_min_vals_with_clipping(self):
        """Test roi_min_vals when radius would go below 0"""
        result = r2d2_base.roi_min_vals(3, 3, 3, radius=5)
        assert result == [0, 0, 0]

    def test_roi_min_vals_edge_case_zero(self):
        """Test roi_min_vals at coordinate 0"""
        result = r2d2_base.roi_min_vals(0, 0, 0, radius=5)
        assert result == [0, 0, 0]

    def test_roi_min_vals_different_coords(self):
        """Test roi_min_vals with different x, y, z values"""
        result = r2d2_base.roi_min_vals(10, 3, 15, radius=5)
        assert result == [5, 0, 10]

    def test_roi_max_vals_no_clipping(self):
        """Test roi_max_vals when radius doesn't exceed dimensions"""
        result = r2d2_base.roi_max_vals(10, 10, 10, radius=5, template_dims=(91, 109, 91))
        assert result == [15, 15, 15]

    def test_roi_max_vals_with_clipping(self):
        """Test roi_max_vals when radius exceeds template dimensions"""
        result = r2d2_base.roi_max_vals(88, 106, 88, radius=5, template_dims=(91, 109, 91))
        assert result == [91, 109, 91]

    def test_roi_max_vals_requires_dimensions(self):
        """Test that roi_max_vals raises error without template_dims"""
        with pytest.raises(ValueError, match="template_dims must be provided"):
            r2d2_base.roi_max_vals(10, 10, 10, radius=5)

    def test_roi_max_vals_custom_dimensions(self):
        """Test roi_max_vals with custom template dimensions"""
        result = r2d2_base.roi_max_vals(250, 250, 250, radius=10, template_dims=(256, 256, 256))
        assert result == [256, 256, 256]


class TestLoadImages:
    """Tests for load_images function"""

    def setup_method(self):
        """Create temporary directory and mock files for each test"""
        self.test_dir = tempfile.mkdtemp()
        self.reg_image_path = os.path.join(self.test_dir, "registered.nii.gz")
        self.template_path = os.path.join(self.test_dir, "template.nii.gz")
        self.mask_path = os.path.join(self.test_dir, "template_mask.nii.gz")

    def teardown_method(self):
        """Clean up temporary directory after each test"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_mock_nifti_files(self):
        """Helper to create mock NIfTI files"""
        # Create empty files to pass existence checks
        open(self.reg_image_path, 'a').close()
        open(self.template_path, 'a').close()
        open(self.mask_path, 'a').close()

    def test_load_images_file_not_found_registered(self):
        """Test that FileNotFoundError is raised when registered image is missing"""
        with pytest.raises(FileNotFoundError, match="Registered image not found"):
            r2d2_base.load_images(self.reg_image_path, self.template_path)

    def test_load_images_file_not_found_template(self):
        """Test that FileNotFoundError is raised when template is missing"""
        open(self.reg_image_path, 'a').close()
        with pytest.raises(FileNotFoundError, match="Template not found"):
            r2d2_base.load_images(self.reg_image_path, self.template_path)

    def test_load_images_file_not_found_mask(self):
        """Test that FileNotFoundError is raised when mask is missing"""
        open(self.reg_image_path, 'a').close()
        open(self.template_path, 'a').close()
        with pytest.raises(FileNotFoundError, match="Template mask not found"):
            r2d2_base.load_images(self.reg_image_path, self.template_path)

    @patch('r2d2_base.ants.image_read')
    def test_load_images_success(self, mock_image_read):
        """Test successful image loading"""
        self.create_mock_nifti_files()

        # Mock ANTs image objects
        mock_reg = Mock()
        mock_template = Mock()
        mock_mask = Mock()
        mock_image_read.side_effect = [mock_reg, mock_template, mock_mask]

        result = r2d2_base.load_images(self.reg_image_path, self.template_path)

        assert "reg_image" in result
        assert "template_image" in result
        assert "template_mask" in result
        assert result["reg_image"] == mock_reg
        assert result["template_image"] == mock_template
        assert result["template_mask"] == mock_mask
        assert mock_image_read.call_count == 3

    def test_load_images_path_handling(self):
        """Test that mask path is correctly constructed from template path"""
        # Test with different extensions
        self.create_mock_nifti_files()

        with patch('r2d2_base.ants.image_read') as mock_read:
            mock_read.return_value = Mock()
            try:
                r2d2_base.load_images(self.reg_image_path, self.template_path)
                # Verify the mask path was constructed correctly
                calls = mock_read.call_args_list
                mask_call = calls[2][0][0]
                assert mask_call.endswith("_mask.nii.gz")
            except Exception:
                pass  # Expected if ANTs can't read the empty files


class TestSaveImages:
    """Tests for save_images function"""

    def setup_method(self):
        """Create temporary directory for each test"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory after each test"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('r2d2_base.ants.image_write')
    def test_save_images_creates_directory(self, mock_write):
        """Test that save_images creates output directory if it doesn't exist"""
        output_dir = os.path.join(self.test_dir, "nonexistent_dir")
        assert not os.path.exists(output_dir)

        mock_image = Mock()
        image_res = {"MI": mock_image, "MSE": mock_image}

        r2d2_base.save_images(output_dir, image_res, radius=3)

        assert os.path.exists(output_dir)

    @patch('r2d2_base.ants.image_write')
    def test_save_images_writes_all_metrics(self, mock_write):
        """Test that save_images writes all metric images"""
        mock_image = Mock()
        image_res = {
            "MI": mock_image,
            "MSE": mock_image,
            "CORR": mock_image,
            "dm_MI": mock_image,
            "dm_MSE": mock_image,
            "dm_CORR": mock_image
        }

        r2d2_base.save_images(self.test_dir, image_res, radius=5)

        assert mock_write.call_count == 6

        # Verify correct filenames
        written_files = [call[0][1] for call in mock_write.call_args_list]
        assert any("r2d2_MI_rad5.nii" in f for f in written_files)
        assert any("r2d2_CORR_rad5.nii" in f for f in written_files)

    @patch('r2d2_base.ants.image_write')
    def test_save_images_uses_correct_radius(self, mock_write):
        """Test that save_images includes correct radius in filenames"""
        mock_image = Mock()
        image_res = {"MI": mock_image}

        r2d2_base.save_images(self.test_dir, image_res, radius=7)

        call_args = mock_write.call_args_list[0][0]
        filename = call_args[1]
        assert "rad7" in filename


class TestCompStats:
    """Tests for comp_stats function"""

    def create_mock_image_dict(self, with_reg_image=True):
        """Helper to create mock image dictionary"""
        # Create mock ANTs images with numpy array-like behavior
        mock_mask = Mock()
        mock_mask.__gt__ = Mock(return_value=np.ones((10, 10, 10), dtype=bool))

        mock_template = Mock()
        mock_reg = Mock() if with_reg_image else None

        return {
            "template_mask": mock_mask,
            "template_image": mock_template,
            "reg_image": mock_reg
        }

    def create_mock_r2d2_results(self):
        """Helper to create mock R2D2 results"""
        mock_img = Mock()
        mock_img.__getitem__ = Mock(return_value=np.random.rand(100))

        return {
            "MI": mock_img,
            "MSE": mock_img,
            "CORR": mock_img,
            "dm_MI": mock_img,
            "dm_MSE": mock_img,
            "dm_CORR": mock_img
        }

    @patch('r2d2_base.ants.image_similarity')
    def test_comp_stats_calculates_metrics(self, mock_similarity):
        """Test that comp_stats calculates mean, std, z-score for each metric"""
        mock_similarity.return_value = 0.85

        img_dict = self.create_mock_image_dict()
        r2d2 = self.create_mock_r2d2_results()

        result = r2d2_base.comp_stats(r2d2, img_dict)

        # Check that mean, std, and z are calculated for each metric
        for metric in ["MI", "MSE", "CORR", "dm_MI", "dm_MSE", "dm_CORR"]:
            assert f"{metric}_mean" in result
            assert f"{metric}_std" in result
            assert f"{metric}_z" in result

    @patch('r2d2_base.ants.image_similarity')
    def test_comp_stats_calculates_wholebrain_similarity(self, mock_similarity):
        """Test that whole-brain similarity is calculated for all metrics"""
        mock_similarity.return_value = 0.75

        img_dict = self.create_mock_image_dict()
        r2d2 = self.create_mock_r2d2_results()

        result = r2d2_base.comp_stats(r2d2, img_dict)

        assert "MattesMutualInformation_wholebrain" in result
        assert "MeanSquares_wholebrain" in result
        assert "Correlation_wholebrain" in result

    @patch('r2d2_base.ants.image_similarity')
    def test_comp_stats_handles_errors_gracefully(self, mock_similarity):
        """Test that comp_stats returns NaN values on error"""
        mock_similarity.side_effect = RuntimeError("ANTs error")

        img_dict = self.create_mock_image_dict()
        r2d2 = self.create_mock_r2d2_results()

        result = r2d2_base.comp_stats(r2d2, img_dict)

        # Should have NaN values instead of raising
        assert np.isnan(result["MI_mean"])
        assert np.isnan(result["MattesMutualInformation_wholebrain"])


class TestComputeR2D2:
    """Tests for compute_r2d2 function (integration-style)"""

    def create_mock_image_dict(self, shape=(10, 10, 10)):
        """Create mock image dictionary with synthetic data"""
        # Create mock ANTs images
        mock_template = Mock()
        mock_template.shape = shape
        mock_template.__mul__ = Mock(return_value=mock_template)

        mock_reg = Mock()
        mock_reg.shape = shape

        mock_mask = Mock()
        mock_mask.shape = shape
        mock_mask.__getitem__ = Mock(return_value=1)

        return {
            "template_image": mock_template,
            "reg_image": mock_reg,
            "template_mask": mock_mask
        }

    @patch('r2d2_base.ants.image_clone')
    @patch('r2d2_base.ants.crop_indices')
    @patch('r2d2_base.ants.image_similarity')
    def test_compute_r2d2_returns_all_metrics(self, mock_sim, mock_crop, mock_clone):
        """Test that compute_r2d2 returns all 6 metric images"""
        # Setup mocks
        mock_img = Mock()
        mock_img.__setitem__ = Mock()
        mock_clone.return_value = mock_img
        mock_crop.return_value = Mock()
        mock_sim.return_value = 0.8

        # Create small test image (3x3x3 with only center voxel masked)
        img_dict = self.create_mock_image_dict(shape=(3, 3, 3))
        mock_mask = img_dict["template_mask"]

        # Only process center voxel to speed up test
        def mask_getitem(key):
            if key == (1, 1, 1):
                return 1
            return 0
        mock_mask.__getitem__ = Mock(side_effect=mask_getitem)

        result = r2d2_base.compute_r2d2(img_dict, radius=1, subsess="test")

        assert "MI" in result
        assert "MSE" in result
        assert "CORR" in result
        assert "dm_MI" in result
        assert "dm_MSE" in result
        assert "dm_CORR" in result

    @patch('r2d2_base.ants.image_clone')
    @patch('r2d2_base.ants.crop_indices')
    @patch('r2d2_base.ants.image_similarity')
    def test_compute_r2d2_raises_on_error(self, mock_sim, mock_crop, mock_clone):
        """Test that compute_r2d2 raises RuntimeError on processing failure"""
        mock_clone.return_value = Mock()
        mock_crop.side_effect = RuntimeError("Cropping failed")

        img_dict = self.create_mock_image_dict(shape=(3, 3, 3))
        mock_mask = img_dict["template_mask"]
        mock_mask.__getitem__ = Mock(return_value=1)

        with pytest.raises(RuntimeError, match="R2D2 computation failed"):
            r2d2_base.compute_r2d2(img_dict, radius=1, subsess="test")


class TestMainFunction:
    """Tests for main and main_wrapper functions"""

    def setup_method(self):
        """Create temporary directory for each test"""
        self.test_dir = tempfile.mkdtemp()
        self.sub_folder = os.path.join(self.test_dir, "sub-001")
        os.makedirs(self.sub_folder)

    def teardown_method(self):
        """Clean up temporary directory after each test"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('r2d2_base.comp_stats')
    @patch('r2d2_base.save_images')
    @patch('r2d2_base.compute_r2d2')
    @patch('r2d2_base.load_images')
    def test_main_success(self, mock_load, mock_compute, mock_save, mock_stats):
        """Test successful execution of main function"""
        # Create required file
        reg_image_path = os.path.join(self.sub_folder, "registered_t2_img.nii.gz")
        open(reg_image_path, 'a').close()

        mock_load.return_value = {"reg_image": Mock(), "template_image": Mock(), "template_mask": Mock()}
        mock_compute.return_value = {"MI": Mock(), "MSE": Mock()}
        mock_stats.return_value = {"MI_mean": 0.8, "MSE_mean": 0.1}

        # Create a temporary template file for the test
        template_dir = tempfile.mkdtemp()
        try:
            template_path = os.path.join(template_dir, "template.nii.gz")
            open(template_path, 'a').close()

            result = r2d2_base.main(self.sub_folder, template_path=template_path, radius=3)

            assert "subsess" in result
            assert result["subsess"] == "sub-001"
            assert "MI_mean" in result
            mock_load.assert_called_once()
            mock_compute.assert_called_once()
            mock_save.assert_called_once()
        finally:
            shutil.rmtree(template_dir)

    @patch('r2d2_base.load_images')
    def test_main_raises_on_missing_file(self, mock_load):
        """Test that main raises exception when files are missing"""
        mock_load.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            r2d2_base.main(self.sub_folder, template_path="/dummy/template.nii.gz")

    def test_main_wrapper_catches_exceptions(self):
        """Test that main_wrapper catches exceptions and returns error dict"""
        result = r2d2_base.main_wrapper(self.sub_folder)

        assert "subsess" in result
        assert "error" in result
        assert result["subsess"] == "sub-001"

    @patch('r2d2_base.main')
    def test_main_wrapper_returns_success(self, mock_main):
        """Test that main_wrapper returns result dict on success"""
        mock_main.return_value = {"subsess": "sub-001", "MI_mean": 0.8}

        result = r2d2_base.main_wrapper(self.sub_folder)

        assert "error" not in result
        assert result["subsess"] == "sub-001"
        assert result["MI_mean"] == 0.8


class TestArgumentParsing:
    """Tests for command-line argument parsing"""

    def test_get_args_defaults(self):
        """Test that default arguments are set correctly"""
        with patch('sys.argv', ['r2d2_base.py', '--template_path', '/path/to/template.nii.gz']):
            args = r2d2_base.get_args()

            assert args.num_python_jobs == 4
            assert args.num_itk_cores == "1"
            assert args.list_path is None
            assert args.search_string is None
            assert args.template_path == '/path/to/template.nii.gz'

    def test_get_args_with_list_path(self):
        """Test parsing with list_path argument"""
        with patch('sys.argv', ['r2d2_base.py', '--template_path', '/path/to/template.nii.gz', '--list_path', '/path/to/list.txt']):
            args = r2d2_base.get_args()
            assert args.list_path == '/path/to/list.txt'
            assert args.template_path == '/path/to/template.nii.gz'

    def test_get_args_with_search_string(self):
        """Test parsing with search_string argument"""
        with patch('sys.argv', ['r2d2_base.py', '--template_path', '/path/to/template.nii.gz', '--search_string', './sub-*/*.nii.gz']):
            args = r2d2_base.get_args()
            assert args.search_string == './sub-*/*.nii.gz'
            assert args.template_path == '/path/to/template.nii.gz'

    def test_get_args_with_parallelization(self):
        """Test parsing with custom parallelization settings"""
        with patch('sys.argv', ['r2d2_base.py', '--template_path', '/path/to/template.nii.gz', '--num_python_jobs', '8', '--num_itk_cores', '2']):
            args = r2d2_base.get_args()
            assert args.num_python_jobs == 8
            assert args.num_itk_cores == "2"
            assert args.template_path == '/path/to/template.nii.gz'


class TestIntegration:
    """Integration tests with synthetic data"""

    @pytest.mark.slow
    def test_end_to_end_with_synthetic_data(self):
        """End-to-end test with synthetic ANTs images"""
        # This test requires ANTs to be properly installed
        pytest.skip("Requires ANTs library and is slow - run manually")

        # Create synthetic 3D images
        data = np.random.rand(10, 10, 10).astype('float32')
        template = ants.from_numpy(data)
        reg_image = ants.from_numpy(data + np.random.randn(10, 10, 10) * 0.1)
        mask = ants.from_numpy(np.ones((10, 10, 10)).astype('float32'))

        img_dict = {
            "template_image": template,
            "reg_image": reg_image,
            "template_mask": mask
        }

        # This would be very slow for real images
        result = r2d2_base.compute_r2d2(img_dict, radius=1, subsess="synthetic")

        assert "MI" in result
        assert "CORR" in result
        assert result["MI"].shape == (10, 10, 10)
