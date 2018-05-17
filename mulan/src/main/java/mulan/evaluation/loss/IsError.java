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

import java.util.ArrayList;
import java.util.List;

/**
 * Implementation of the IsError loss function, which is simply the indicator
 * of whether the induced ranking is perfect or not. Speaking in terms of error
 * set, the loss is zero if the cardinality of the error-set is zero
 * and one if the cardinality of the error set is greather than zero.
 * 
 * @author Grigorios Tsoumakas
 * @version 2010.11.10
 */
public class IsError extends RankingLossFunctionBase {

    public String getName() {
        return "IsError";
    }

    @Override
    public double computeLoss(int[] ranking, boolean[] groundTruth) {
        List<Integer> relevant = new ArrayList<Integer>();
        List<Integer> irrelevant = new ArrayList<Integer>();
        int numLabels = groundTruth.length;
        for (int index = 0; index < numLabels; index++) {
            if (groundTruth[index]) {
                relevant.add(index);
            } else {
                irrelevant.add(index);
            }
        }

        double isError = 0;
        boolean terminate = false;
        for (int rLabel : relevant) {
            for (int irLabel : irrelevant) {
                if (ranking[rLabel] > ranking[irLabel]) {
                    isError = 1;
                    terminate = true;
                    break;
                }
            }
            if (terminate) {
                break;
            }
        }

        return isError;
    }
}