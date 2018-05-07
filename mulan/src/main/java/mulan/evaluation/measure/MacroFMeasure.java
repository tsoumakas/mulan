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
package mulan.evaluation.measure;

/**
 * Implementation of the macro-averaged f measure.
 *
 * @author Grigorios Tsoumakas
 * @version 2012.05.29
 */
public class MacroFMeasure extends LabelBasedFMeasure implements MacroAverageMeasure {

    /**
     * Constructs a new object with given number of labels and beta=1
     *
     * @param numOfLabels the number of labels
     */
    public MacroFMeasure(int numOfLabels) {
        this(numOfLabels, 1);
    }

    /**
     * Full constructor
     *
     * @param numOfLabels the number of labels
     * @param beta controls the combination of precision and recall
     */
    public MacroFMeasure(int numOfLabels, double beta) {
        super(numOfLabels, beta);
    }

    @Override
    public String getName() {
        return "Macro-averaged F-Measure";
    }

    @Override
    public double getValue() {
        double sum = 0;
        int count = 0;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            sum += InformationRetrievalMeasures.fMeasure(truePositives[labelIndex],
                    falsePositives[labelIndex],
                    falseNegatives[labelIndex], beta);
            count++;
        }
        return sum / count;
    }

    /**
     * Returns the F-Measure for a label
     *
     * @param labelIndex the index of a label (starting from 0)
     * @return the F-Measure for the given label
     */
    @Override
    public double getValue(int labelIndex) {
        return InformationRetrievalMeasures.fMeasure(truePositives[labelIndex],
                falsePositives[labelIndex],
                falseNegatives[labelIndex], beta);
    }

}