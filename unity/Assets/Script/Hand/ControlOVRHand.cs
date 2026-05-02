using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ControlOVRHand : MonoBehaviour
{
    OVRHand leftHand;
    // Start is called before the first frame update
    void Start()
    {
        leftHand = transform.GetComponent<OVRHand>();
        leftHand.enabled = false;
        leftHand.gameObject.SetActive(true); // 确保手部模型始终显示
        // leftHand.EnableHandTracking = false;
        // leftHand. = false; 
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
