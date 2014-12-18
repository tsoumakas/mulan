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

/*
 *    LabelSet.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import java.io.Serializable;
import java.util.ArrayList;
import weka.core.Utils;

/**
 * Class that handles labelsets <p>
 *
 * @author Grigorios Tsoumakas
 * @author Robert Friberg
 * @version $Revision: 0.03 $
 */
public class LabelSet implements Serializable, Comparable<LabelSet> {

    private static final long serialVersionUID = 7658871740607834759L;
    /**
     * The set is represented internally as an array of integers.
     * Weka uses doubles but we choose integers for fast comparisons.
     * Observe that the set is immutable,
     * once assigned by the constructor, no changes are possible.
     */
    protected int[] labelSet;

    /**
     * Initializes an object based on an array of doubles containing 0/1
     *
     * @param set array of doubles containing 0 and 1
     */
    public LabelSet(double[] set) {
        labelSet = new int[set.length];
        for (int i = 0; i < set.length; i++) {
            labelSet[i] = (int) set[i];
        }
    }

    /**
     * A comma-separated list of label names enclosed in curlies.
     */
    @Override
    public String toString() {
        return toBitString();
    }

    @Override
    public int hashCode() {
        return toString().hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof LabelSet) {
            LabelSet other = (LabelSet) obj;
            return other.labelSet.length == this.labelSet.length && hammingDifference(other) == 0;
        } else {
            return false; //could perhaps allow comparison with double array
        }
    }
    /**
     * A cached count of the set size. Observe that the set
     * size is not the same as the size of the double array.
     */
    private int size = -1;

    /**
     * The number of set members. Calculated on first call and
     * cached for subsequent calls.
     * @return The number of set members
     */
    public int size() {
        if (size == -1) {
            size = Utils.sum(labelSet);
        }

        return size;
    }

    /**
     * Get an array representation of this set.
     * @return a copy of the underlying array.
     */
    public double[] toDoubleArray() {
        double[] arr = new double[labelSet.length];
        for (int i = 0; i < labelSet.length; i++) {
            arr[i] = labelSet[i];
        }
        return arr;
    }

    /**
     * Get an array representation of this set.
     * @return a copy of the underlying array.
     */
    public boolean[] toBooleanArray() {
        boolean[] arr = new boolean[labelSet.length];
        for (int i = 0; i < labelSet.length; i++) {
            arr[i] = (labelSet[i] == 1) ? true : false;
        }
        return arr;
    }

    /**
     * Calculates the Hamming Distance between the current labelset and another labelset.
     *
     * @param other the other LabelSet object.
     * @return the Hamming Distance.
     */
    public int hammingDifference(LabelSet other) {
        int diff = 0;
        for (int i = 0; i < labelSet.length; i++) {
            if (labelSet[i] != other.labelSet[i]) {
                diff++;
            }
        }
        return diff;
    }

    /**
     * Constructs a bitstring from the current labelset.
     * @return the bitstring.
     */
    public String toBitString() {
        StringBuilder sb = new StringBuilder(labelSet.length);
        for (int i = 0; i < labelSet.length; i++) {
            sb.append(Integer.toString(labelSet[i]));
        }
        return sb.toString();
    }

    /**
     * Constructs a LabelSet object from a bitstring.
     *
     * @param bits the bitstring
     * @return the labelset.
     * @throws Exception if creation fails due to invalid bitstring
     */
    public static LabelSet fromBitString(String bits) throws Exception {
        LabelSet result = new LabelSet(new double[bits.length()]);
        for (int i = 0; i < bits.length(); i++) {
            switch (bits.charAt(i)) {
                case '1':
                    result.labelSet[i] = 1;
                    break;
                case '0':
                    result.labelSet[i] = 0;
                    break;
                default:
                    throw new Exception("Bad bitstring: " + bits);
            }
        }
        return result;
    }

    /**
     * Constructs all subsets of a labelset (apart from the empty one).
     *
     * @return an ArrayList of LabelSet objects with the subsets.
     * @throws Exception if creation of labelset's subsets fails
     */
    public ArrayList<LabelSet> getSubsets() throws Exception {
        ArrayList<LabelSet> subsets = new ArrayList<LabelSet>();

        //the number of members of a power set is 2^n
        int powerElements = (int) Math.pow(2, size());

        for (int i = 1; i < powerElements - 1; i++) {
            String temp = Integer.toBinaryString(i);
            int foundDigits = temp.length();
            for (int j = foundDigits; j < size(); j++) {
                temp = "0" + temp;
            }
            LabelSet tempSet = LabelSet.fromBitString(temp);
            double[] tempDouble = tempSet.toDoubleArray();

            double[] subset = new double[labelSet.length];
            int counter = 0;
            for (int j = 0; j < labelSet.length; j++) {
                if (labelSet[j] == 0) {
                    subset[j] = 0;
                } else {
                    subset[j] = tempDouble[counter];
                    counter++;
                }
            }

            LabelSet finalLabelSet = new LabelSet(subset);
            subsets.add(finalLabelSet);
        }
        return subsets;
    }

    /**
     *
     * @param l1 a labelset
     * @param l2 another labelset
     * @return their interesection
     */
    public static LabelSet intersection(LabelSet l1, LabelSet l2) {
        double[] arrayL1 = l1.toDoubleArray();
        double[] arrayL2 = l2.toDoubleArray();

        if (arrayL1.length != arrayL2.length) {
            return null;
        }

        double[] intersection = new double[arrayL2.length];
        for (int i = 0; i < arrayL2.length; i++) {
            if (arrayL1[i] == 1 && arrayL2[i] == 1) {
                intersection[i] = 1;
            } else {
                intersection[i] = 0;
            }
        }

        return new LabelSet(intersection);
    }

    /**
     * Used for sorting collections of labelsets according to size
     */
    public int compareTo(LabelSet o) {
        if (o.size() < size()) {
            return -1;
        } else if (o.size() > size()) {
            return 1;
        } else {
            return 0;
        }
    }
}
