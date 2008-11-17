package mulan.core;

import java.util.Random;

/**
 * Class which provides various utility methods.
 * 
 * @author Jozef Vilcek
 */
public class Util {

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

		if (counter == 1)
			return maxIndexes[0];
		else {
			int choose = rand.nextInt(counter);
			return maxIndexes[choose];
		}
	}
}
