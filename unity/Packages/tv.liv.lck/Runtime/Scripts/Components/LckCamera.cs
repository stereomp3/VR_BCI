using UnityEngine;

#if LCK_UNITY_URP
using UnityEngine.Rendering.Universal;
#endif

namespace Liv.Lck
{
    [RequireComponent(typeof(Camera))]
    public class LckCamera : MonoBehaviour, ILckCamera
    {
        [SerializeField]
        private Camera _camera;

        [SerializeField]
        private string _cameraId;

        public string CameraId => _cameraId;

        private void Awake()
        {
            if (string.IsNullOrEmpty(_cameraId))
            {
                _cameraId = System.Guid.NewGuid().ToString();
            }

            _camera.enabled = false;
            LckMediator.RegisterCamera(this);
            
#if LCK_UNITY_URP
            LckLog.Log($"Configuring URP camera data for camera: {_cameraId}");
            if (!_camera.TryGetComponent(out UniversalAdditionalCameraData urpCameraData))
                urpCameraData = _camera.gameObject.AddComponent<UniversalAdditionalCameraData>();
            
            urpCameraData.allowXRRendering = false;
#endif
        }

        private void OnDestroy()
        {
            LckMediator.UnregisterCamera(this);
        }

        public void ActivateCamera(RenderTexture renderTexture)
        {
            _camera.enabled = true;
            _camera.targetTexture = renderTexture;
        }

        public void DeactivateCamera()
        {
            _camera.enabled = false;
            _camera.targetTexture = null;
        }

        public Camera GetCameraComponent()
        {
            return _camera;
        }
    }
}
