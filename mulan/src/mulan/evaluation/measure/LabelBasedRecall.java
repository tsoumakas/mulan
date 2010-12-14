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
 *    LabelBasedRecall.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.evaluation.measure;

/**
 * Common class for the micro/macro label-based recall measures.
 * 
 * @author Grigorios Tsoumakas
 * @version 2010.12.10
 */
public abstract class LabelBasedRecall extends LabelBasedBipartitionMeasureBase {

    /**
     * Constructs a new object with given number of labels
     *
     * @param numOfLabels the number of labels
     */
    public LabelBasedRecall(int numOfLabels) {
        super(numOfLabels);
    }

    public double getIdealValue() {
        return 1;
    }

    protected void updateBipartition(boolean[] bipartition, boolean[] truth) {
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
            boolean actual = truth[labelIndex];
            boolean predicted = bipartition[labelIndex];

            if (actual && predicted) {
                truePositives[labelIndex]++;
            }
            if (actual && !predicted) {
                falseNegatives[labelIndex]++;
            }
        }
    }
}
