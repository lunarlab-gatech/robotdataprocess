import numpy as np
import os
import unittest
from pathlib import Path
from robotdataprocess.data_types.TransformationData import TransformationData
from robotdataprocess.data_types.Data import CoordinateFrame, TransformType
from scipy.spatial.transform import Rotation as R

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestTransformationData(unittest.TestCase):

    def test_from_HERCULES_settings_json(self):
        """ Test loading transformation from HERCULES settings JSON with a non-identity rotation. """
        json_path = Path(Path('.'), 'tests', 'files', 'test_TransformationData', 'test_from_HERCULES_settings_json', 'settings.json').absolute()
        
        # Load the transformation data for Drone1, front_center camera (has a Pitch of 10 degrees)
        transformation_data = TransformationData.from_HERCULES_settings_json(
            json_path=str(json_path),
            robot_name="Drone1", 
            sensor_type="camera",
            sensor_name="front_center"
        )

        # Verify basic attributes
        self.assertEqual(transformation_data.frame_id, "Drone1")
        self.assertEqual(transformation_data.child_frame_id, "front_center")
        self.assertEqual(transformation_data.frame, CoordinateFrame.NED)

        # Define the expected Ground Truth matrix for Drone1, front_center camera
        # Translation: X=0.35, Y=-0.055, Z=0.2
        # Rotation: Roll=0, Pitch=10, Yaw=0
        expected_translation_values = np.array([0.35, -0.055, 0.2])
        expected_rotation_matrix = R.from_euler(seq="xyz", angles=[0.0, 10.0, 0.0], degrees=True,).as_matrix()
        
        expected_matrix = np.identity(4)
        expected_matrix[0:3, 0:3] = expected_rotation_matrix
        expected_matrix[0:3, 3] = expected_translation_values

        # Get the transformation matrix from the loaded data
        actual_matrix = transformation_data.as_matrix()

        np.testing.assert_array_almost_equal(actual_matrix, expected_matrix)
    
    def test_from_matrix(self):
        """ Test creating TransformationData from a 4x4 matrix, including a non-identity rotation. """
        # Matrix with a rotation (90 deg around Z) and translation
        matrix_to_test = np.array([
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        tf_data = TransformationData.from_matrix("test_frame", "child_test_frame", matrix_to_test, CoordinateFrame.FLU)

        self.assertEqual(tf_data.frame_id, "test_frame")
        self.assertEqual(tf_data.child_frame_id, "child_test_frame")
        self.assertEqual(tf_data.frame, CoordinateFrame.FLU)
        np.testing.assert_array_almost_equal(tf_data.translation, np.array([10.0, 20.0, 30.0]))
        
        # Manually calculate expected quaternion from rotation matrix part
        expected_rotation_quat = R.from_matrix(matrix_to_test[0:3, 0:3]).as_quat()
        np.testing.assert_array_almost_equal(tf_data.orientation, expected_rotation_quat)

        # Verify as_matrix() returns the original matrix
        np.testing.assert_array_almost_equal(tf_data.as_matrix(), matrix_to_test)

        # Test invalid matrix shape
        with self.assertRaises(ValueError):
            TransformationData.from_matrix("test", "child", np.identity(3), CoordinateFrame.FLU)

    def test_to_coordinate_frame(self):
        """ Test converting transformation data from NED to FLU with an initial non-identity rotation. """
        # Initial transformation in NED frame with 90-degree Yaw rotation and translation
        initial_translation = np.array([1.0, 2.0, 3.0])
        initial_rotation_quat = R.from_euler('z', 90, degrees=True).as_quat() # [0,0,0.707,0.707] xyzw
        
        initial_matrix = np.identity(4)
        initial_matrix[0:3, 0:3] = R.from_quat(initial_rotation_quat).as_matrix()
        initial_matrix[0:3, 3] = initial_translation

        transformation_data = TransformationData.from_matrix(
            frame_id="robot",
            child_frame_id="sensor",
            matrix=initial_matrix,
            frame=CoordinateFrame.NED
        )

        # Test no-op if target_frame is current_frame — returns a copy
        same_frame_result = transformation_data.to_coordinate_frame(CoordinateFrame.NED)
        self.assertEqual(same_frame_result.frame, CoordinateFrame.NED)
        np.testing.assert_array_almost_equal(same_frame_result.as_matrix(), initial_matrix)
        # Original should be unchanged
        np.testing.assert_array_almost_equal(transformation_data.as_matrix(), initial_matrix)


        # Test NED to FLU conversion
        flu_result = transformation_data.to_coordinate_frame(CoordinateFrame.FLU)
        self.assertEqual(flu_result.frame, CoordinateFrame.FLU)

        # Original should be unchanged (still NED)
        self.assertEqual(transformation_data.frame, CoordinateFrame.NED)
        np.testing.assert_array_almost_equal(transformation_data.as_matrix(), initial_matrix)

        # Expected translation after NED to FLU (Y, Z flipped signs)
        expected_translation_after_transform = np.array([1.0, -2.0, -3.0])

        # Expected orientation after NED to FLU (initial_rotation_quat * q_NED_to_FLU)
        # q_NED_to_FLU is 180 deg around X: [1,0,0,0] xyzw
        # initial_rotation_quat is 90 deg around Z: [0,0,0.707,0.707] xyzw
        q_NED_to_FLU_transform = R.from_euler('x', 180, degrees=True)
        expected_orientation_after_transform = (q_NED_to_FLU_transform * R.from_quat(initial_rotation_quat)).as_quat() # [0.0, 0.70710678, 0.70710678, 0.0] xyzw

        np.testing.assert_array_almost_equal(flu_result.translation, expected_translation_after_transform)
        np.testing.assert_array_almost_equal(flu_result.orientation, expected_orientation_after_transform)

        # Verify the full matrix after transformation
        expected_matrix_after_transform = np.identity(4)
        expected_matrix_after_transform[0:3, 0:3] = R.from_quat(expected_orientation_after_transform).as_matrix()
        expected_matrix_after_transform[0:3, 3] = expected_translation_after_transform
        np.testing.assert_array_almost_equal(flu_result.as_matrix(), expected_matrix_after_transform)


        # Test unsupported conversion (e.g., FLU to NED) - flu_result is FLU, so converting back to NED is not implemented
        with self.assertRaises(NotImplementedError):
            flu_result.to_coordinate_frame(CoordinateFrame.NED)

        # Test unsupported conversion (e.g., initial NED to ENU)
        unsupported_initial_matrix = np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        unsupported_data = TransformationData.from_matrix(
            frame_id="robot",
            child_frame_id="sensor",
            matrix=unsupported_initial_matrix,
            frame=CoordinateFrame.NED
        )
        with self.assertRaises(NotImplementedError):
            unsupported_data.to_coordinate_frame(CoordinateFrame.ENU)

    def test_to_coordinate_frame_change_of_basis(self):
        """ Test CHANGE_OF_BASIS mode for to_coordinate_frame (NED -> FLU). """
        # Build a transformation with rotation (90 deg yaw) and translation in NED
        initial_translation = np.array([1.0, 2.0, 3.0])
        initial_rotation_quat = R.from_euler('z', 90, degrees=True).as_quat()

        initial_matrix = np.identity(4)
        initial_matrix[0:3, 0:3] = R.from_quat(initial_rotation_quat).as_matrix()
        initial_matrix[0:3, 3] = initial_translation

        tf_ned = TransformationData.from_matrix("robot", "sensor", initial_matrix, CoordinateFrame.NED)

        # --- No-op when target == current frame ---
        same = tf_ned.to_coordinate_frame(CoordinateFrame.NED, TransformType.CHANGE_OF_BASIS)
        np.testing.assert_array_almost_equal(same.as_matrix(), initial_matrix)

        # --- Change of basis NED -> FLU ---
        tf_flu = tf_ned.to_coordinate_frame(CoordinateFrame.FLU, TransformType.CHANGE_OF_BASIS)
        self.assertEqual(tf_flu.frame, CoordinateFrame.FLU)

        # Compute expected: T_new = R * T * R^{-1}, R = Rx(180)
        R_frame = R.from_euler('x', 180, degrees=True).as_matrix()
        R_4x4 = np.identity(4)
        R_4x4[0:3, 0:3] = R_frame
        R_inv_4x4 = np.identity(4)
        R_inv_4x4[0:3, 0:3] = R_frame.T
        expected_matrix = R_4x4 @ initial_matrix @ R_inv_4x4

        np.testing.assert_array_almost_equal(tf_flu.as_matrix(), expected_matrix)

        # --- Verify ROTATION mode produces correct values ---
        tf_flu_rotation = tf_ned.to_coordinate_frame(CoordinateFrame.FLU, TransformType.ROTATION)
        # Expected rotation result: q_new = Rx(180) * q_old, t_new = [1, -2, -3]
        expected_rotation_translation = np.array([1.0, -2.0, -3.0])
        q_NED_to_FLU = R.from_euler('x', 180, degrees=True)
        expected_rotation_orientation = (q_NED_to_FLU * R.from_quat(initial_rotation_quat)).as_quat()
        expected_rotation_matrix = np.identity(4)
        expected_rotation_matrix[0:3, 0:3] = R.from_quat(expected_rotation_orientation).as_matrix()
        expected_rotation_matrix[0:3, 3] = expected_rotation_translation
        np.testing.assert_array_almost_equal(tf_flu_rotation.as_matrix(), expected_rotation_matrix)

        # The two modes should NOT be identical in general
        self.assertFalse(np.allclose(tf_flu.as_matrix(), tf_flu_rotation.as_matrix()),
                         "CHANGE_OF_BASIS and ROTATION should produce different results for this transformation")

        # --- Verify original is unchanged ---
        np.testing.assert_array_almost_equal(tf_ned.as_matrix(), initial_matrix)

        # --- Property: change of basis preserves eigenvalues of the rotation ---
        orig_eigenvalues = np.sort(np.linalg.eigvals(initial_matrix[0:3, 0:3]))
        new_eigenvalues = np.sort(np.linalg.eigvals(tf_flu.as_matrix()[0:3, 0:3]))
        np.testing.assert_array_almost_equal(np.abs(orig_eigenvalues), np.abs(new_eigenvalues))

        # --- Identity transformation should remain identity under change of basis ---
        tf_identity = TransformationData.from_matrix("A", "B", np.identity(4), CoordinateFrame.NED)
        tf_identity_cob = tf_identity.to_coordinate_frame(CoordinateFrame.FLU, TransformType.CHANGE_OF_BASIS)
        np.testing.assert_array_almost_equal(tf_identity_cob.as_matrix(), np.identity(4))

    def test_apply_transformation_right_side(self):
        """ Test applying a transformation to the right side. """
        # tf_A_B: A to B
        tf_A_B_matrix = np.array([
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        tf_A_B = TransformationData.from_matrix("frame_A", "frame_B", tf_A_B_matrix, CoordinateFrame.FLU)

        # tf_B_C: B to C
        tf_B_C_matrix = np.array([
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, -1.0, 6.0],
            [0.0, 1.0, 0.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        tf_B_C = TransformationData.from_matrix("frame_B", "frame_C", tf_B_C_matrix, CoordinateFrame.FLU)

        # Expected result: tf_A_C = tf_A_B @ tf_B_C
        expected_tf_A_C_matrix = tf_A_B_matrix @ tf_B_C_matrix

        # Apply the transformation
        result_tf = tf_A_B.apply_transformation_right_side(tf_B_C)

        self.assertEqual(result_tf.frame_id, "frame_A")
        self.assertEqual(result_tf.child_frame_id, "frame_C")
        self.assertEqual(result_tf.frame, CoordinateFrame.FLU)
        np.testing.assert_array_almost_equal(result_tf.as_matrix(), expected_tf_A_C_matrix)

        # Ensure original tf_A_B is not modified
        np.testing.assert_array_almost_equal(tf_A_B.as_matrix(), tf_A_B_matrix)
        self.assertEqual(tf_A_B.child_frame_id, "frame_B")


        # Test incompatible coordinate frames
        tf_A_B_ned = TransformationData.from_matrix("frame_A", "frame_B", tf_A_B_matrix, CoordinateFrame.NED)
        tf_B_C_flu = TransformationData.from_matrix("frame_B", "frame_C", tf_B_C_matrix, CoordinateFrame.FLU)
        with self.assertRaises(ValueError):
            tf_A_B_ned.apply_transformation_right_side(tf_B_C_flu)
        
        # Test incompatible frame_id and child_frame_id
        tf_A_B_compatible_frame = TransformationData.from_matrix("frame_A", "frame_B", tf_A_B_matrix, CoordinateFrame.FLU)
        tf_D_C_incompatible_id = TransformationData.from_matrix("frame_D", "frame_C", tf_B_C_matrix, CoordinateFrame.FLU)
        with self.assertRaises(ValueError):
            tf_A_B_compatible_frame.apply_transformation_right_side(tf_D_C_incompatible_id)

    def test_invert(self):
        """ Test that T @ T.invert() == Identity for a non-trivial transformation. """
        # Build a transformation with rotation (90 deg around Z) and translation
        matrix = np.array([
            [0.0, -1.0, 0.0, 10.0],
            [1.0,  0.0, 0.0, 20.0],
            [0.0,  0.0, 1.0, 30.0],
            [0.0,  0.0, 0.0,  1.0],
        ])
        tf = TransformationData.from_matrix("frame_A", "frame_B", matrix, CoordinateFrame.FLU)

        tf_inv = tf.invert()

        # Frame IDs should be swapped
        self.assertEqual(tf_inv.frame_id, "frame_B")
        self.assertEqual(tf_inv.child_frame_id, "frame_A")
        self.assertEqual(tf_inv.frame, CoordinateFrame.FLU)

        # T @ T_inv should equal the identity matrix
        product = tf.as_matrix() @ tf_inv.as_matrix()
        np.testing.assert_array_almost_equal(product, np.identity(4))

        # T_inv @ T should also equal the identity matrix
        product_reverse = tf_inv.as_matrix() @ tf.as_matrix()
        np.testing.assert_array_almost_equal(product_reverse, np.identity(4))

        # Also verify with a more complex rotation (Euler xyz 30, 45, 60)
        rot = R.from_euler('xyz', [30, 45, 60], degrees=True)
        complex_matrix = np.identity(4)
        complex_matrix[0:3, 0:3] = rot.as_matrix()
        complex_matrix[0:3, 3] = [5.0, -3.0, 7.5]
        tf2 = TransformationData.from_matrix("world", "sensor", complex_matrix, CoordinateFrame.NED)

        tf2_inv = tf2.invert()
        np.testing.assert_array_almost_equal(tf2.as_matrix() @ tf2_inv.as_matrix(), np.identity(4))
        np.testing.assert_array_almost_equal(tf2_inv.as_matrix() @ tf2.as_matrix(), np.identity(4))

        # Identity transformation should invert to itself
        tf_identity = TransformationData.from_matrix("A", "B", np.identity(4), CoordinateFrame.FLU)
        tf_identity_inv = tf_identity.invert()
        np.testing.assert_array_almost_equal(tf_identity_inv.as_matrix(), np.identity(4))

    def test_OpenVINS_transformations(self):

        # Define test sequences and GT transforms
        dataset_names = ["V2.3.AP", "V2.3.AC"]
        robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]

        gt_trans = {
            "V2.3.AP": {
                "Husky1": np.array([[ 0.   ,  0.  ,   1.  ,  0.    ],
                                    [ 1.   ,  0.  ,   0.  ,  -0.055],
                                    [ 0.   ,  1.  ,   0.  ,  -0.85 ],
                                    [ 0.   ,  0.  ,   0.  ,   1.   ]]),
                "Husky2": np.array([[ 0.   ,  0.  ,   1.  ,  0.    ],
                                    [ 1.   ,  0.  ,   0.  ,  -0.055],
                                    [ 0.   ,  1.  ,   0.  ,  -0.85 ],
                                    [ 0.   ,  0.  ,   0.  ,   1.   ]]),
                "Drone1": np.array([[0.0, 0.17364817766693033, 0.984807753012208, 0.35],
                                    [1.0, 0.0, 0.0, -0.055],
                                    [0.0, 0.984807753012208, -0.17364817766693033, 0.2],
                                    [0.0, 0.0, 0.0, 1.0]]),
                "Drone2": np.array([[0.0, 0.17364817766693033, 0.984807753012208, 0.35],
                                    [1.0, 0.0, 0.0, -0.055],
                                    [0.0, 0.984807753012208, -0.17364817766693033, 0.2],
                                    [0.0, 0.0, 0.0, 1.0]]),
            },
            "V2.3.AC": {
                "Husky1": np.array([[ 0.   ,  0.  ,   1.  ,  0.    ],
                                    [ 1.   ,  0.  ,   0.  ,  -0.055],
                                    [ 0.   ,  1.  ,   0.  ,  -0.85 ],
                                    [ 0.   ,  0.  ,   0.  ,   1.   ]]),
                "Husky2": np.array([[ 0.   ,  0.  ,   1.  ,  0.    ],
                                    [ 1.   ,  0.  ,   0.  ,  -0.055],
                                    [ 0.   ,  1.  ,   0.  ,  -0.85 ],
                                    [ 0.   ,  0.  ,   0.  ,   1.   ]]),
                "Drone1": np.array([[0.0, 0.17364817766693033, 0.984807753012208, 0.35],
                                    [1.0, 0.0, 0.0, -0.055],
                                    [0.0, 0.984807753012208, -0.17364817766693033, 0.2],
                                    [0.0, 0.0, 0.0, 1.0]]),
                "Drone2": np.array([[0.0, 0.17364817766693033, 0.984807753012208, 0.35],
                                    [1.0, 0.0, 0.0, -0.055],
                                    [0.0, 0.984807753012208, -0.17364817766693033, 0.2],
                                    [0.0, 0.0, 0.0, 1.0]]),
            }
        }

        for dataset_name in dataset_names:
            for robot_name in robot_names:
                
                # Load the configuration from HERCULES
                json_path = Path(Path('.'), 'tests', 'files', 'test_TransformationData', 'test_OpenVINS_transformations', 'settings_' + dataset_name + '.json').absolute()
                H_R_to_C = TransformationData.from_HERCULES_settings_json(str(json_path), robot_name, "camera", "stereo_left")

                # Calculate pose of optical frame with respect to Robot (IMU)
                H_C_to_O = TransformationData.optical_wrt_camera(CoordinateFrame.NED, "stereo_left", "stereo_left_optical")
                H_R_to_O = H_R_to_C.apply_transformation_right_side(H_C_to_O)

                # Compare with GT
                np.testing.assert_array_almost_equal(H_R_to_O.as_matrix(), gt_trans[dataset_name][robot_name], decimal=12)
    
    def test_ROMAN_transformations(self):

        # Define test sequences and GT transforms
        dataset_names = ["V2.4.C", "V2.4.F"]
        robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]

        gt_trans = {
            "V2.4.F": {
                "Husky1": np.array([[ 0.0,  0.0, 1.0, 0.0],
                                    [-1.0,  0.0, 0.0, 0.055],
                                    [ 0.0, -1.0, 0.0, 0.85],
                                    [ 0.0,  0.0, 0.0, 1.0]]),
                "Husky2": np.array([[ 0.0,  0.0, 1.0, 0.0],
                                    [-1.0,  0.0, 0.0, 0.055],
                                    [ 0.0, -1.0, 0.0, 0.85],
                                    [ 0.0,  0.0, 0.0, 1.0]]),
                "Drone1": np.array([[ 0.0, 0.17364817766693033, 0.984807753012208, 0.35],
                                    [-1.0, 0.0, 0.0, 0.055],
                                    [0.0, -0.984807753012208, 0.17364817766693033, -0.2],
                                    [0.0, 0.0, 0.0, 1.0]]),
                "Drone2": np.array([[ 0.0, 0.17364817766693033, 0.984807753012208, 0.35],
                                    [-1.0, 0.0, 0.0, 0.055],
                                    [0.0, -0.984807753012208, 0.17364817766693033, -0.2],
                                    [0.0, 0.0, 0.0, 1.0]]),
            },
            "V2.4.C": {
                "Husky1": np.array([[ 0.0, 0.0, 1.0, 0.0],
                                    [-1.0, 0.0, 0.0, 0.055],
                                    [0.0, -1.0, 0.0, 0.85],
                                    [0.0, 0.0, 0.0, 1.0]]),
                "Husky2": np.array([[ 0.0, 0.0, 1.0, 0.0],
                                    [-1.0, 0.0, 0.0, 0.055],
                                    [0.0, -1.0, 0.0, 0.85],
                                    [0.0, 0.0, 0.0, 1.0]]),
                "Drone1": np.array([[ 0.0, 0.17364817766693033, 0.984807753012208, 0.35],
                                    [-1.0, 0.0, 0.0, 0.055],
                                    [0.0, -0.984807753012208, 0.17364817766693033, -0.2],
                                    [0.0, 0.0, 0.0, 1.0]]),
                "Drone2": np.array([[ 0.0, 0.17364817766693033, 0.984807753012208, 0.35],
                                    [-1.0, 0.0, 0.0, 0.055],
                                    [0.0, -0.984807753012208, 0.17364817766693033, -0.2],
                                    [0.0, 0.0, 0.0, 1.0]]),
            }
        }

        for dataset_name in dataset_names:
            for robot_name in robot_names:
                
                # Load the configuration from HERCULES
                json_path = Path(Path('.'), 'tests', 'files', 'test_TransformationData', 
                                 'test_ROMAN_transformations', 'settings_' + dataset_name + '.json').absolute()
                H_R_to_C = TransformationData.from_HERCULES_settings_json(str(json_path), robot_name, "camera", "stereo_left")

                # Calculate pose of optical frame with respect to Robot (IMU)
                H_C_to_O = TransformationData.optical_wrt_camera(CoordinateFrame.NED, "stereo_left", "stereo_left_optical")
                H_R_to_O = H_R_to_C.apply_transformation_right_side(H_C_to_O)

                # Convert from NED to FLU frame
                H_R_to_O_FLU = H_R_to_O.to_coordinate_frame(CoordinateFrame.FLU)

                # Compare with GT
                np.testing.assert_array_almost_equal(H_R_to_O_FLU.as_matrix(), gt_trans[dataset_name][robot_name], decimal=12)


if __name__ == "__main__":
    unittest.main()
