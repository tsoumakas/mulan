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
 * Implementation of the macro-averaged precision measure.
 * 
 * @author Grigorios Tsoumakas
 * @version 2010.11.05
 */
public class MacroPrecision extends LabelBasedPrecision implements MacroAverageMeasure {

    /**
     * Constructs a new object with given number of labels
     * 
     * @param numOfLabels the number of labels
     */
    public MacroPrecision(int numOfLabels) {
        super(numOfLabels);
    }

    @Override
    public double getValue() {
        double sum = 0;
        int count = 0;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            sum += InformationRetrievalMeasures.precision(truePositives[labelIndex], falsePositives[labelIndex], falseNegatives[labelIndex]);
            count++;
        }
        return sum / count;
    }

    @Override
    public String getName() {
        return "Macro-averaged Precision";
    }

    /**
     * Returns the precision for a label
     *
     * @param labelIndex the index of a label (starting from 0)
     * @return the precision for the given label
     */
    @Override
    public double getValue(int labelIndex) {
        return InformationRetrievalMeasures.precision(truePositives[labelIndex],
                falsePositives[labelIndex],
                falseNegatives[labelIndex]);
    }

}