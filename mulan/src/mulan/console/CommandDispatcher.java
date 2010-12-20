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

import java.util.Set;

/**
 * Dispatch the mulan console commands to appropriate specialized handlers. 
 * This is a root handler for the console interface.
 * 
 * @author Jozef Vilcek
 */
public class CommandDispatcher {

	private static String LEARN_EXP_CMD = "learn";
	private static String PREDICT_EXP_CMD = "predict";
	
	/**
	 * Main entry to launch command dispatcher.
	 * 
	 * @param args arguments to execute with
	 */
	public static void main(String[] args) {
		if(args.length == 0 || args[0] == null || args[0].equals("")){
			printGeneralHelp();
		} else if(args[0].equals(LEARN_EXP_CMD)){
			String learner = args.length > 1 ?
					(args[1].startsWith("-") ? null : args[1]) : null;
			processLearnCommand(learner, args);
		} else if(args[0].equals(PREDICT_EXP_CMD)){
			processPredictCommand(args);
		}
		else{
			printGeneralHelp();
		}
	}
	
	private static void printGeneralHelp(){
		System.out.println("usage: mulan <command> [options]");
		System.out.println("type: 'mulan <command> -h' for the help on a specific command");
		System.out.println();
		System.out.println("Available commands:");
		System.out.println("\t" + LEARN_EXP_CMD + ":\t\tperforms learning experiment on the specific learner");
		System.out.println("\t" + PREDICT_EXP_CMD + ":\tperforms prediction on test data from stored learner state");
	}
	
	private static void printLearnHelp(){
		System.out.println("usage: mulan learn <learner_name> [options]");
		System.out.println("type: 'mulan learn --list' to list all names of the available learners");
		System.out.println("type: 'mulan learn  <learner_name> --help' for help on particular learner usage");
	}
	
	private static void processLearnCommand(String learnerName, String[] args) {
		if(learnerName == null || learnerName.equals("")){
			// try process listing if requested or print help
			if(args.length >= 2 && args[1] != null && args[1].equals("--list")){
				System.out.println("Available learners:");
		        LearningDriver driver = new LearningDriver();
		        Set<String> learners = driver.getSupportedLearners();
		        for(String learner : learners){
		        	System.out.println("\t" + learner);
		        }
			}
        	else{
		       	printLearnHelp();
		    }
		} else {
			// process learning via driver
			LearningDriver driver = new LearningDriver();
			driver.doLearningExperiment(learnerName, args);
		}
	}
	
	private static void processPredictCommand(String[] args) {
		PredictionDriver driver = new PredictionDriver();
		driver.doPredictionExperiment(args);
	}

}
