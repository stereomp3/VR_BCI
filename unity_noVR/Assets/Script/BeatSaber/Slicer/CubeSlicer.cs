using UnityEngine;
using EzySlice;

public class CubeSlicer : MonoBehaviour
{
    public Vector3 nor_pos;
    public Material crossSectionMaterial; // Assign in the Inspector

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            SliceCube();
        }
    }

    void SliceCube()
    {
        // Define the slicing plane
        Vector3 planeNormal = nor_pos.normalized; // Adjust as needed
        Vector3 planePosition = transform.position;

        // Perform the slice
        SlicedHull slicedHull = gameObject.Slice(planePosition, planeNormal, crossSectionMaterial);

        if (slicedHull != null)
        {
            // Create upper and lower hulls
            GameObject upperHull = slicedHull.CreateUpperHull(gameObject, crossSectionMaterial);
            GameObject lowerHull = slicedHull.CreateLowerHull(gameObject, crossSectionMaterial);

            // Optional: Add components or set properties
            upperHull.AddComponent<MeshCollider>().convex = true;
            lowerHull.AddComponent<MeshCollider>().convex = true;

            // Destroy the original object
            Destroy(gameObject);
        }
    }
}
