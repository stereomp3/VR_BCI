using UnityEngine;
using EzySlice;
using System.Collections;

public class SaberSlicer : MonoBehaviour
{
    public Material crossSectionMaterial; // Assign in Inspector
    public ParticleSystem cut_effect;
    private Vector3 entryPoint;
    private Vector3 exitPoint;
    private bool isInsideCube = false;
    private GameManager GM;
    private Vector3 face_dir;
    private void Start()
    {
        GM = GameManager.instance;
        face_dir = new Vector3(0, 0, 1);
    }
    private void OnTriggerEnter(Collider other)
    { 
        if (other.CompareTag("Sliceable"))  // 之後要分左右邊 R B 和砍的方向
        {
            entryPoint = transform.position;
            isInsideCube =true;
            StartCoroutine(StartCut(other.gameObject, 0.01f));
            GM.correct_slice += 1; // tmp
        }
    }

    /*private void OnTriggerExit(Collider other) // 要等出來有點慢
    {
        if (other.CompareTag("Sliceable")) //  && isInsideCube
        {
            exitPoint = transform.position;
            Instantiate(cut_effect, other.transform.position, Quaternion.identity); // spawn cut effect
            SliceCube(other.gameObject);
            ScoreManager.instance.add_score(100);
            isInsideCube = false;
        }
    }*/

    void SliceCube(GameObject target)
    {
        Vector3 sliceDirection = exitPoint - entryPoint;
        Debug.Log("@@@@@@@@@@@@ sliceDirection: " + sliceDirection); // -0.2x left, 0.2x right, -0.2y down, 0.2y up
        Vector3 planeNormal = Vector3.Cross(sliceDirection, face_dir).normalized; // Camera.main.transform.forward
        Vector3 planePosition = (entryPoint + exitPoint) / 2f;
        SlicedHull slicedHull = target.Slice(planePosition, planeNormal, crossSectionMaterial);
        if (slicedHull == null) // 修復平面，切割位置設為中心
        {
            Vector3 fallbackPlanePos = target.transform.position;  // 中心點
            slicedHull = target.Slice(fallbackPlanePos, planeNormal, crossSectionMaterial);
            if (slicedHull == null) Debug.LogError("Fallback slice also failed.");
        }
        Destroy(target);
        if (GM.setLogCallback != null) GM.setLogCallback.Invoke(0, LogType.Cut); // 紀錄 log 到 log.txt
        if (slicedHull != null)
        {
            GameObject upperHull = slicedHull.CreateUpperHull(target, crossSectionMaterial);
            GameObject lowerHull = slicedHull.CreateLowerHull(target, crossSectionMaterial);
            upperHull.transform.position = target.transform.position;
            lowerHull.transform.position = target.transform.position;

            Rigidbody upperRb = upperHull.AddComponent<Rigidbody>();
            Rigidbody lowerRb = lowerHull.AddComponent<Rigidbody>();

            /*MeshCollider upperCollider = upperHull.AddComponent<MeshCollider>();
            upperCollider.convex = true;

            MeshCollider lowerCollider = lowerHull.AddComponent<MeshCollider>();
            lowerCollider.convex = true;*/

            // Add force in "slice up/down" direction
            Vector3 pushDir = Vector3.Cross(sliceDirection, planeNormal).normalized;

            float forceMagnitude = 2f;
            upperRb.AddForce(pushDir * forceMagnitude, ForceMode.Impulse);
            lowerRb.AddForce(-pushDir * forceMagnitude, ForceMode.Impulse);

            // Destroy after 1 second
            StartCoroutine(FadeOutAndDestroy(upperHull, 2f));
            StartCoroutine(FadeOutAndDestroy(lowerHull, 2f));
        }
    }

    IEnumerator FadeOutAndDestroy(GameObject obj, float duration)
    {
        float elapsed = 0f;
        Vector3 initialScale = obj.transform.localScale;

        while (elapsed < duration)
        {
            float t = elapsed / duration;
            obj.transform.localScale = Vector3.Lerp(initialScale, Vector3.zero, t);
            elapsed += Time.deltaTime;
            yield return null;
        }

        Destroy(obj);
    }
    IEnumerator StartCut(GameObject other, float wait_time)
    {
        yield return new WaitForSeconds(wait_time);

        exitPoint = transform.position;
        Instantiate(cut_effect, other.transform.position, Quaternion.identity); // spawn cut effect
        SliceCube(other.gameObject);
    }
}