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
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.console;

import org.junit.Test;

public class CommandDispatcherTest {
	
	@Test
	public void test(){
		//String[] args = new String[]{"learn", "bpmll", "-trd=./data/testData/emotions-train.arff", "-tsd=./data/testData/emotions-test.arff", "-ld=./data/testData/emotions.xml"};
		String[] args = new String[]{"learn", "mulan.classifier.neural.BPMLL", "--help"};
		CommandDispatcher.main(args);
	}
}
