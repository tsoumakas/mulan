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
package mulan.classifier.meta.thresholding;

import mulan.classifier.meta.MultiLabelMetaLearnerTest;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.evaluation.measure.HammingLoss;
import weka.classifiers.trees.J48;

public class SCutTest extends MultiLabelMetaLearnerTest {

    @Override
    public void setUp() throws Exception {
        learner = new SCut(new CalibratedLabelRanking(new J48()), new HammingLoss(), 5);
    }
}

