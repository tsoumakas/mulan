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
 *    MMPRandomizedUpdateRule.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural;

import java.util.*;
import mulan.classifier.neural.model.Neuron;
import mulan.evaluation.loss.RankingLossFunction;

/**
 * Implementation of randomized update rule for {@link MMPLearner}. It is a randomized variation 
 * of {@link MMPUniformUpdateRule}. A opposed to uniformed update, the randomized version will
 * penalize each wrongly order pair of labels by random value from interval &lt;0,1&gt;. Afterwards, the 
 * penalty weights are normalized, so their sum is equal to 1.<br>
 * The model is represented as a list of perceptrons (one for each label), each represented 
 * by {@link Neuron}. Perceptrons are expected to be in the same order as labels in training data set.
 * 
 * @see MMPUpdateRuleBase
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
public class MMPRandomizedUpdateRule extends MMPUpdateRuleBase {

    /**
     * Creates a new instance of {@link MMPRandomizedUpdateRule}.
     *
     * @param perceptrons the list of perceptrons, representing the model, which will receive updates.
     * @param lossMeasure the loss measure used to decide when the model should be updated by the rule
     */
    public MMPRandomizedUpdateRule(List<Neuron> perceptrons, RankingLossFunction lossMeasure) {
        super(perceptrons, lossMeasure);
    }

    @Override
    protected double[] computeUpdateParameters(DataPair example, double[] confidences, double loss) {
        int numLabels = example.getOutput().length;

        boolean[] trueOutput = example.getOutputBoolean();

        Set<Integer> relevant = new HashSet<Integer>();
        Set<Integer> irrelevant = new HashSet<Integer>();
        for (int index = 0; index < numLabels; index++) {
            if (trueOutput[index]) {
                relevant.add(index);
            } else {
                irrelevant.add(index);
            }
        }

        // discover wrongly ordered pairs of labels and assign a random weight.
        //	we keep collecting the sum as it will be later used for normalization
        double weightsSum = 0;
        Map<int[], Double> weightedErrorSet = new HashMap<int[], Double>();
        Random rnd = new Random();
        for (int rLabel : relevant) {
            for (int irLabel : irrelevant) {
                if (confidences[rLabel] <= confidences[irLabel]) {
                    double weight = rnd.nextDouble();
                    weightsSum += weight;
                    weightedErrorSet.put(new int[]{rLabel, irLabel}, weight);
                }
            }
        }

        // normalize weights so they all sum to 1 and compute update parameters for perceptrons
        double[] params = new double[numLabels];
        Set<int[]> labelPairs = weightedErrorSet.keySet();
        for (int[] pair : labelPairs) {
            double weight = weightedErrorSet.get(pair);
            params[pair[0]] += weight * loss;
            params[pair[1]] -= weight * loss;
        }

        return params;
    }
}