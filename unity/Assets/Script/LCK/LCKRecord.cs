using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Liv.Lck.Recorder;
using Liv.Lck.Smoothing;
using Liv.Lck.UI;

namespace Liv.Lck.Tablet
{
    public class LCKRecord : MonoBehaviour
    {
        public bool is_record = false;
        public bool stop_record = false;
        public LCKCameraController LCK;
        // Start is called before the first frame update
        void Start()
        {
            stop_record = false;
            if (is_record == true)
            {
                LCK.ToggleRecording();
            }
        }

        // Update is called once per frame
        void Update()
        {
            if(stop_record == true)
            {
                LCK.ToggleRecording();
                stop_record = false;
            }
        }
    }

}
