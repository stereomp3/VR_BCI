namespace Liv.Lck
{
    using UnityEngine;

    public static class LckRigidbodyExtensions
    {
        /// <summary>
        /// Rotates and positions the ridigbody to look in the given forward direction, about the pivot point provided.
        /// </summary>
        /// <param name="rigidbody">The rigidbody to modify</param>
        /// <param name="pivot">The pivot point to rotate around</param>
        /// <param name="forward">The desired forward direction to look in</param>
        /// <param name="position">The Current position</param>
        /// <param name="currentRotation">The current rotation of the rigidbody</param>
        public static void LookAtFromPivotPoint(
            this Rigidbody rigidbody,
            Vector3 pivot,
            Vector3 forward,
            Vector3 position,
            Quaternion currentRotation)
        {
            var lookAtRotation = Quaternion.LookRotation(
                forward.normalized,
                Vector3.up);
            
            var deltaRotationToLookAt = lookAtRotation * Quaternion.Inverse(currentRotation);
            var relativeRootPosition = position - pivot;
            var rotatedRelativePosition = deltaRotationToLookAt * relativeRootPosition;
            var newRootPosition = pivot + rotatedRelativePosition;
            rigidbody.MovePosition(newRootPosition);
            rigidbody.MoveRotation(lookAtRotation);
        }
    }
}