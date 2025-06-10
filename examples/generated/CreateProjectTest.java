package org.eclipse.swtbot.test;

import static org.junit.Assert.*;
import org.eclipse.swtbot.swt.finder.SWTBot;
import org.eclipse.swtbot.eclipse.finder.SWTWorkbenchBot;
import org.eclipse.swtbot.swt.finder.junit.SWTBotJunit4ClassRunner;
import org.eclipse.swtbot.swt.finder.utils.SWTBotPreferences;
import org.eclipse.swtbot.swt.finder.widgets.*;
import org.eclipse.swtbot.swt.finder.exceptions.WidgetNotFoundException;
import org.eclipse.swtbot.swt.finder.waits.Conditions;
import org.eclipse.swtbot.swt.finder.waits.DefaultCondition;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Auto-generated test from AI Test Script Generator
 * Test creating a new project
 */
@RunWith(SWTBotJunit4ClassRunner.class)
public class CreateProjectTest {
    
    private SWTWorkbenchBot bot;
    
    @Before
    public void setUp() throws Exception {
        // Configure SWTBot preferences
        SWTBotPreferences.PLAYBACK_DELAY = 100;
        SWTBotPreferences.TIMEOUT = 10000;
        
        // Initialize bot
        bot = new SWTWorkbenchBot();
        
        // Close welcome view if it exists
        try {
            bot.viewByTitle("Welcome").close();
        } catch (WidgetNotFoundException e) {
            // Welcome view not found, ignore
        }
    }
    
    @Test
    public void createProject() throws Exception {
        // Click on 'File' menu
        bot.menu("File").click();
        
        // Click on 'New' submenu
        bot.menu("File").menu("New").click();
        
        // Click on 'Project' menu item
        bot.menu("File").menu("New").menu("Project").click();
        
        // Wait for New Project dialog
        bot.waitUntil(Conditions.shellIsActive("New Project"));
        
        // Select 'Java Project' from the wizard
        bot.tree().select("Java Project");
        bot.button("Next").click();
        
        // Enter 'TestProject' as the project name
        bot.textWithLabel("Project name:").setText("TestProject");
        
        // Click on 'Finish' button
        bot.button("Finish").click();
        
        // Wait for project creation
        bot.waitUntil(new DefaultCondition() {
            @Override
            public boolean test() {
                try {
                    return bot.viewByTitle("Project Explorer").bot().tree().getTreeItem("TestProject").isVisible();
                } catch (Exception e) {
                    return false;
                }
            }
            
            @Override
            public String getFailureMessage() {
                return "Project did not appear in Project Explorer within timeout";
            }
        });
        
        // Verify that project appears in Project Explorer
        assertTrue("Project should appear in Project Explorer",
            bot.viewByTitle("Project Explorer").bot().tree().getTreeItem("TestProject").isVisible());
    }
    
    @After
    public void tearDown() throws Exception {
        // Clean up any open dialogs
        while (tryCloseShell());
    }
    
    /**
     * Helper method to close any open dialogs
     */
    private boolean tryCloseShell() {
        try {
            SWTBotShell shell = bot.activeShell();
            if (shell != null && shell.isOpen() && !shell.isActive()) {
                shell.close();
                return true;
            }
        } catch (Exception e) {
            // Ignore exceptions
        }
        return false;
    }
}
