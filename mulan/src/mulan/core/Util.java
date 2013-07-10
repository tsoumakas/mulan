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
package mulan.core;

import java.util.Random;

/**
 * Class which provides various utility methods.
 * 
 * @author Jozef Vilcek
 */
public class Util {

    /** Constant representing a new line */
    private static final String NEW_LINE = System.getProperty("line.separator");

    /**
     * Procedure to find index of maximum value in the specified array.
     * If there is only one unique maximum, the index of this maximum is returned.
     * If there are more values equal to the maximum, one of these values is randomly
     * selected and its index is returned.
     *
     * @param array the array in which the maximum value should be find
     * @param rand random instance used to select the value if more values equal
     *             to the maximum are present in the array
     * @return the index of find maximum value
     */
    public static int RandomIndexOfMax(double array[], Random rand) {

        int[] maxIndexes = new int[array.length];
        double max = array[0];
        maxIndexes[0] = 0;
        int counter = 1;

        for (int i = 1; i < array.length; i++) {
            if (array[i] == max) {
                maxIndexes[counter] = i;
                counter++;
            } else if (array[i] > max) {
                max = array[i];
                maxIndexes[0] = i;
                counter = 1;
            }
        }

        if (counter == 1) {
            return maxIndexes[0];
        } else {
            int choose = rand.nextInt(counter);
            return maxIndexes[choose];
        }
    }

    /**
     * Returns a correct new line separator string for the underlying operating system.
     * @return the new line separator string
     */
    public static String getNewLineSeparator() {
        return NEW_LINE;
    }
}