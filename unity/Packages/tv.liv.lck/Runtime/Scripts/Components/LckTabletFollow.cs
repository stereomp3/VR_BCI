using System;
using Liv.Lck.Tablet;
using Liv.Lck.UI;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace Liv.Lck
{
    public class LckTabletFollow : MonoBehaviour
    {
        [Header("Settings")]
        
        [SerializeField]
        private float _heightOffsetForPlayerHead;
        [SerializeField]
        private float _minFollowSmoothing = 0.2f;
        [SerializeField]
        private float _minFollowDistanceMultiplier = 0.75f;

        [Header("References")]
        [SerializeField]
        private LCKCameraController _controller;
        
        [SerializeField]
        private Toggle _isFollowingToggle;
        
        [SerializeField]
        private Transform _selfieCamera;        
        
        [SerializeField]
        [Tooltip("If none, HmdTransform + offset will be used.")]
        private Transform _followTarget;
        
        [SerializeField]
        private LckDoubleButton _smoothingDoubleButton;
        
        [SerializeField]
        private LckDoubleButton _followDistanceDoubleButton;

        [SerializeField]
        private Rigidbody _rigidbodyRoot;

        // Follow toggle is only shown in Selfie mode, so using that as the correct mode
        private bool _isInCorrectCameraMode = true;
        private bool _isFollowToggleOn;
        private Vector3 _followVelocity;
        private Vector3 _targetPosition;
        private Vector3 _offsetBetweenSelfieCameraAndTablet;
        
        // These values are set by their initial DoubleButton value settings
        private float _minFollowDistance;
        private float _followSmoothing;

        private RigidbodyInterpolation _defaultInterpolation;
        
        #region UNITY METHODS

        private void OnEnable()
        {
            _isFollowToggleOn = _isFollowingToggle.isOn;
            _isFollowingToggle.onValueChanged.AddListener(OnIsFollowToggled);
            _followDistanceDoubleButton.OnValueChanged += OnFollowDistanceChanged;
            _smoothingDoubleButton.OnValueChanged += OnSmoothingChanged;
            _controller.OnCameraModeChanged += OnCameraModeChanged;
        }      

        private void OnDisable()
        {
            _isFollowingToggle.onValueChanged.RemoveListener(OnIsFollowToggled);
            _followDistanceDoubleButton.OnValueChanged -= OnFollowDistanceChanged;
            _smoothingDoubleButton.OnValueChanged -= OnSmoothingChanged;
            _controller.OnCameraModeChanged -= OnCameraModeChanged;
        }

        private void SetInitialValuesFromDoubleButtons()
        {
            _minFollowDistance = _followDistanceDoubleButton.Value * _minFollowDistanceMultiplier;
            _followSmoothing = CalculateFollowSmoothing(_smoothingDoubleButton.Value);
        }

        private void Start()
        {
            SetInitialValuesFromDoubleButtons();

            _isInCorrectCameraMode = true;
            _targetPosition = transform.position;
            
            if (_rigidbodyRoot != null)
                _defaultInterpolation = _rigidbodyRoot.interpolation;
        }      

        private void FixedUpdate()
        {
            ProcessTabletFollowingWithRigidbody();
        }

        #endregion
        
        #region PUBLIC METHODS

        public void SetFollowTarget(Transform target)
        {
            _followTarget = target;
        }
        
        #endregion
        
        #region PRIVATE METHODS

        private void ProcessTabletFollowingWithRigidbody()
        {
            if (_isInCorrectCameraMode == false) return;
            if (_isFollowToggleOn == false) return;

            var headPos = !_followTarget ? _controller.HmdTransform.position + Vector3.down * _heightOffsetForPlayerHead : _followTarget.position;

            var tabletPos = _rigidbodyRoot.position;
            var dirFromHeadToTablet = tabletPos - headPos;

            var isClose = dirFromHeadToTablet.magnitude < _minFollowDistance;

            _targetPosition = headPos + dirFromHeadToTablet.normalized * _minFollowDistance;

            var smoothTargetPos = Vector3.SmoothDamp(
                tabletPos,
                isClose ? tabletPos : _targetPosition,
                ref _followVelocity,
                _followSmoothing);
            
            _rigidbodyRoot.MovePosition(smoothTargetPos);

            var selfieCameraPosition = _selfieCamera.transform.position;
           
            _rigidbodyRoot.LookAtFromPivotPoint(
                selfieCameraPosition,
                headPos - selfieCameraPosition,
                smoothTargetPos,
                _rigidbodyRoot.rotation);
        }
        
        private void OnCameraModeChanged(CameraMode mode)
        {
            if (mode == CameraMode.Selfie)
            {
                _isInCorrectCameraMode = true;
            }
            else
            {
                _isInCorrectCameraMode = false;
            }
        }

        private void OnIsFollowToggled(bool value)
        {
            _isFollowToggleOn = value;
            
            if(_rigidbodyRoot != null)
                _rigidbodyRoot.interpolation = _isFollowToggleOn ? RigidbodyInterpolation.Interpolate : _defaultInterpolation;
        }

        private void OnSmoothingChanged(float value)
        {
            _followSmoothing = CalculateFollowSmoothing(value);
        }

        private float CalculateFollowSmoothing(float value)
        {
            if (value / 10f < _minFollowSmoothing)
            {
                return _minFollowSmoothing;
            }
            else
            {
                return value / 10f;
            }
        }

        private void OnFollowDistanceChanged(float value)
        {
            _minFollowDistance = value * _minFollowDistanceMultiplier;
        }
        #endregion
    }
}
