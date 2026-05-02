using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Timer 
{
    public static double GetUnixTimestamp()
    {
        DateTime now = DateTime.UtcNow;
        DateTime epochStart = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        return (now - epochStart).TotalSeconds;
    }
}
