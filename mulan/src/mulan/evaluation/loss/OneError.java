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
package mulan.evaluation.loss;

/**
 * Implementation of the one-error loss function. For a given example and
 * prediction, one-error is 1 if the top ranked label is a relevant and 0
 * otherwise.
 * 
 * @author Jozef Vilcek
 * @author Grigorios Tsoumakas
 * @version 2010.11.10
 */
public class OneError extends RankingLossFunctionBase {

    public String getName() {
        return "OneError";
    }

    @Override
    public double computeLoss(int[] ranking, boolean[] groundTruth) {
        double oneError = 0;
        int numLabels = groundTruth.length;
        for (int topRated = 0; topRated < numLabels; topRated++) {
            if (ranking[topRated] == 1) {
                if (!groundTruth[topRated]) {
                    oneError++;
                }
                break;
            }
        }
        return oneError;
    }
}