using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PerpendicularVector : MonoBehaviour
{
    public Transform targetTransform;
    public Vector3 upAxis = Vector3.up;
    private Vector3 previousPosition;
    private Vector3 speed;
    // Start is called before the first frame update
    void Start()
    {
        if (targetTransform != null)
        {
            previousPosition = targetTransform.position;
        }

    }

    // Update is called once per frame
    void Update()
    {
        if (targetTransform != null)
        {
            // Calculate the speed of the target Transform
            speed = (targetTransform.position - previousPosition) / Time.deltaTime;
            previousPosition = targetTransform.position;
            // Calculate a new vector perpendicular to the speed and upAxis
            Vector3 perpendicularVector = Vector3.Cross(speed, upAxis).normalized;
            // Use the perpendicularVector as needed (e.g., Debug.Log, assign to a variable, etc.)
            Debug.Log("Perpendicular Vector: " + perpendicularVector);
        }
    }
}
