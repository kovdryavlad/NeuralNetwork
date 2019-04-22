using SimpleMatrix;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace NeuralNetworkBL
{
    public static class LettersInformator
    {
        static List<char> letters = new List<char>(new[] { 'в', 'л', 'а', 'д' });
        //List<char> letters = new List<char>();

        public static Vector GetVectorForLetter(char letter)
        {
            int index = Array.IndexOf(letters.ToArray(), letter);
            double[] arr = new double[letters.Count];
            arr[index] = 1;

            return new Vector(arr);
        }
        
        public static char GetLetterByVector(Vector vector)
        {
            int maxElementIndex = vector.GetCloneOfData()
                                         .Select((val, index) => new { Value = val, Index = index })
                                         .OrderByDescending(el => el.Value)
                                         .First()
                                         .Index;

            return letters[maxElementIndex];
        }
    }
}