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
 *    MMPMaxUpdateRule.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural;

import java.util.List;
import mulan.classifier.neural.model.Neuron;
import mulan.evaluation.loss.RankingLossFunction;

/**
 * Implementation of max update rule for {@link MMPLearner}. Only two perceptrons will
 * receive updates, the one corresponding to the lowest ranked relevant label and the
 * one corresponding to the highest ranked non-relevant label. <br>
 * The model is represented as a list of perceptrons (one for each label), each represented 
 * by {@link Neuron}. Perceptrons are expected to be in the same order as labels in training 
 * data set.
 * 
 * @see MMPUpdateRuleBase
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
public class MMPMaxUpdateRule extends MMPUpdateRuleBase {

    /**
     * Creates a new instance of {@link MMPMaxUpdateRule}.
     *
     * @param perceptrons the list of perceptrons, representing the model, which will receive updates.
     * @param lossMeasure the loss measure used to decide when the model should be updated by the rule
     */
    public MMPMaxUpdateRule(List<Neuron> perceptrons, RankingLossFunction lossMeasure) {
        super(perceptrons, lossMeasure);
    }

    @Override
    protected double[] computeUpdateParameters(DataPair example, double[] confidences, double loss) {

        int numLabels = example.getOutput().length;
        boolean[] trueOutput = example.getOutputBoolean();
        int lrLabel = -1; // lowest ranked relevant label
        int hirLabel = -1; // highest ranked non-relevant label
        for (int index = 0; index < numLabels; index++) {
            if (trueOutput[index]) {
                if (lrLabel == -1) {
                    lrLabel = index;
                }
                if (confidences[index] <= confidences[lrLabel]) {
                    lrLabel = index;
                }
            } else {
                if (hirLabel == -1) {
                    hirLabel = index;
                }
                if (confidences[index] >= confidences[hirLabel]) {
                    hirLabel = index;
                }
            }
        }

        double[] params = new double[numLabels];
        params[lrLabel] = loss;
        params[hirLabel] = -loss;

        return params;
    }
}