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

/**
 * Implementation of the "ranking loss" ranking loss function. It is basically
 * the size of the error set divided by all possible pairs of relevant and
 * irrelevant labels
 * 
 * @author Grigorios Tsoumakas
 * @version 2010.11.05
 */
public class RankingLoss extends ErrorSetSize {

    @Override
    public String getName() {
        return "Ranking Loss";
    }

    @Override
    public double computeLoss(int[] ranking, boolean[] groundTruth) {
        int numLabels = groundTruth.length;
        ArrayList<Integer> trueIndexes = new ArrayList<Integer>();
        ArrayList<Integer> falseIndexes = new ArrayList<Integer>();
        for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
            if (groundTruth[labelIndex]) {
                trueIndexes.add(labelIndex);
            } else {
                falseIndexes.add(labelIndex);
            }
        }

        if (!trueIndexes.isEmpty() && !falseIndexes.isEmpty()) {
            int rolp = 0; // reversed ordered label pairs
            for (int k : trueIndexes) {
                for (int l : falseIndexes) {
                    //	if (output[instanceIndex].getConfidences()[trueIndexes.get(k)] <= output[instanceIndex].getConfidences()[falseIndexes.get(l)])
                    if (ranking[k] > ranking[l]) {
                        rolp++;
                    }
                }
            }
            return (double) rolp / (trueIndexes.size() * falseIndexes.size());
        } else {
            return 0;
        }
    }
}