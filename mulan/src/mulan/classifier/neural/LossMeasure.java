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
*    LossMeasure.java
*    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
*
*/

package mulan.classifier.neural;


/**
 * The enumeration of loss measures, which can be used by the 
 * {@link MMPLearner} to determine the ranking performance 
 * during the learning phase.
 * 
 * @author Jozef Vilcek
 */
public enum LossMeasure {
	/** Indicates usage of One-Error Measure */
	OneError,
	/** Indicates usage of Error Set Size Measure (Ranking Loss Measure) */
	ErrorSetSize,
	/** Indicates usage of Is-Error Measure */
	IsError,
	/** Indicates usage of Average Precision Measure */
	AveragePrecision
}
