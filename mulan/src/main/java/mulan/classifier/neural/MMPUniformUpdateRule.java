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
 *    MMPUniformUpdateRule.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import mulan.classifier.neural.model.Neuron;
import mulan.evaluation.loss.RankingLossFunction;

/**
 * Implementation of uniform update rule for {@link MMPLearner}. The rule will penalize 
 * each wrongly order pair of labels by same amount. <br>
 * The model is represented as a list of perceptrons (one for each label), each represented 
 * by {@link Neuron}. Perceptrons are expected to be in the same order as labels in training data set.
 * 
 * @see MMPUpdateRuleBase
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
public class MMPUniformUpdateRule extends MMPUpdateRuleBase {

    /**
     * Creates a new instance of {@link MMPUniformUpdateRule}.
     *
     * @param perceptrons the list of perceptrons, representing the model, which will receive updates.
     * @param lossMeasure the loss measure used to decide when the model should be updated by the rule
     */
    public MMPUniformUpdateRule(List<Neuron> perceptrons, RankingLossFunction lossMeasure) {
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

        // discover wrongly ordered pairs of labels and count number of positive
        // (relevant label need higher rank) and negative (irrelevant label needs
        // lower rank) hits for each label ... these will be turned  to update
        // parameters for update (shift) of particular perceptrons.
        int setCount = 0;
        double[] params = new double[numLabels];
        for (int rLabel : relevant) {
            for (int irLabel : irrelevant) {
                if (confidences[rLabel] <= confidences[irLabel]) {
                    params[rLabel]++;
                    params[irLabel]--;
                    setCount++;
                }
            }
        }

        for (int index = 0; index < numLabels; index++) {
            if (params[index] != 0) {
                params[index] *= loss / setCount;
            }
        }

        return params;
    }
}