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
 * Implementation of the ErrorSetSize loss function, which computes the size of
 * the error set. The error set is composed of all possible label pairs,
 * where one is relevant and the other is not, and which satisfy the condition
 * that the relevant label is ranked lower than the irrelevant one.
 * 
 * @author Grigorios Tsoumakas
 * @version 2010.11.10
 */
public class ErrorSetSize extends RankingLossFunctionBase {

    public String getName() {
        return "ErrorSetSize";
    }

    @Override
    public double computeLoss(int[] ranking, boolean[] groundTruth) {
        double ess = 0;
        int numLabels = groundTruth.length;
        List<Integer> relevant = new ArrayList<Integer>();
        List<Integer> irrelevant = new ArrayList<Integer>();
        for (int index = 0; index < numLabels; index++) {
            if (groundTruth[index]) {
                relevant.add(index);
            } else {
                irrelevant.add(index);
            }
        }

        for (int rLabel : relevant) {
            for (int irLabel : irrelevant) {
                if (ranking[rLabel] > ranking[irLabel]) {
                    ess++;
                }
            }
        }

        return ess;
    }
}