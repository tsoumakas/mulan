package mulan;

/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

import java.io.Serializable;
import weka.core.Utils;

/**
 * Class that handles labelsets <p>
 *
 * @author Grigorios Tsoumakas 
 * @author Robert Friberg
 * @version $Revision: 0.02 $ 
 */
public class LabelSet implements Serializable
{
    private static final long serialVersionUID = 7658871740607834759L;

    /**
     * The set is represented internally as an array of integers.
     * Weka uses doubles but we choose integers for fast comparisons.
     * Observe that the set is immutable,
     * once assigned by the constructor, no changes are possible.
     */
    protected int[] labelSet;

	
    public LabelSet(double[] set)
    {
        labelSet = new int[set.length];
        for(int i = 0; i < set.length; i++) labelSet[i] = (int) set[i];
    }

    /**
     * A comma-separated list of label names enclosed in curlies.
     */
    public String toString()
    {
        return toBitString();
    }	

    public int hashCode()
    {
        return toString().hashCode();
    }

    public boolean equals(Object obj)
    {
        if (obj instanceof LabelSet)
        {
            LabelSet other = (LabelSet) obj;
            return other.labelSet.length == this.labelSet.length
                    && hammingDifference(other) == 0;
        }
        else return false; //could perhaps allow comparison with double array 
    }	
	
    /**
     * A cached count of the set size. Observe that the set
     * size is not the same as the size of the double array.
     */
    private int size = -1;
	
    /**
     * The number of set members. Calculated on first call and
     * cached for subsequent calls.
     * @return
     */
    public int size()
    {
        if (size == -1) 
            size = Utils.sum(labelSet);
        
        return size;
    }

    /**
     * Get an array representation of this set.
     * @return a copy of the underlying array.
     */
    public double[] toDoubleArray()
    {
        double[] arr = new double[labelSet.length];
        for(int i = 0; i < labelSet.length; i++) 
            arr[i] = labelSet[i];
        return arr;
    }
	
    /**
     * calculates the Hamming Distance between the current labelset and another labelset.
     * @return the Hamming Distance.
     */    
    public int hammingDifference(LabelSet other) 
    {
        int diff = 0;
        for(int i = 0; i < labelSet.length; i++)
                if (labelSet[i] != other.labelSet[i]) diff++;
        return diff;
    }

    /**
     * Constructs a bitstring from the current labelset.
     * @return the bitstring.
     */    
    public String toBitString()
    {
        StringBuilder sb = new StringBuilder(labelSet.length);
        for(int i = 0; i < labelSet.length; i++) 
            sb.append(Integer.toString(labelSet[i]));
        return sb.toString();
    }
	        
    /**
     * Constructs a LabelSet object from a bitstring.
     * @return the labelset.
     */    
    public static LabelSet fromBitString(String bits) throws Exception
    {
        LabelSet result = new LabelSet(new double[bits.length()]);
        for(int i = 0; i < bits.length(); i++)
        {
            switch (bits.charAt(i))
            {
                case '1' : result.labelSet[i] = 1;  break;
                case '0' : result.labelSet[i] = 0;  break;
                default: throw new Exception("Bad bitstring: " + bits);
            }
        }
        return result;
    }
}
