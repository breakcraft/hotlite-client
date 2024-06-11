package hotlite.plugins.reinforcementLearning

import com.google.inject.Inject
import com.typesafe.config.ConfigFactory
import kotlinx.coroutines.*
import net.runelite.api.*
import net.runelite.api.coords.WorldPoint
import net.runelite.api.events.*
import net.runelite.client.eventbus.Subscribe
import net.runelite.client.plugins.Plugin
import net.runelite.client.plugins.PluginDescriptor
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.slf4j.LoggerFactory
import java.nio.file.Paths

@PluginDescriptor(
    name = "Reinforcement Learning Plugin",
    description = "Integrates reinforcement learning models into the RuneLite client",
    tags = ["reinforcement", "learning", "osrs"]
)
class ReinforcementLearningPlugin : Plugin() {

    @Inject
    private lateinit var client: Client

    private lateinit var model: Module
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val logger = LoggerFactory.getLogger(ReinforcementLearningPlugin::class.java)
    private val config = ConfigFactory.load()

    private val actionMap = mapOf(
        0 to "attack",
        1 to "defend",
        2 to "move_north",
        3 to "move_south",
        4 to "move_east",
        5 to "move_west",
        6 to "idle"
    )

    override fun startUp() {
        scope.launch {
            try {
                model = loadModelFromConfig()
                setupListeners()
                setupCommands()
                logger.info("Reinforcement Learning Plugin started successfully.")
            } catch (e: Exception) {
                logger.error("Failed to start Reinforcement Learning Plugin", e)
            }
        }
    }

    override fun shutDown() {
        scope.cancel()
        logger.info("Reinforcement Learning Plugin stopped.")
    }

    private fun loadModel(modelPath: String): Module {
        return Module.load(Paths.get(modelPath)).apply {
            logger.info("Model loaded from $modelPath")
        }
    }

    private fun reloadModel(modelPath: String): Boolean {
        return try {
            model = loadModel(modelPath)
            logger.info("Model reloaded from $modelPath")
            true
        } catch (e: Exception) {
            logger.error("Failed to reload model", e)
            false
        }
    }

    private fun loadModelFromConfig(): Module {
        val modelPath = config.getString("rl.model.path")
        return loadModel(modelPath)
    }

    private fun setupListeners() {
        client.addChatMessageListener { event ->
            scope.launch {
                handleEvent { predictAction(event.message) }
            }
        }

        client.addGameTickListener { event ->
            scope.launch {
                handleEvent { predictAction(getCurrentGameState()) }
            }
        }

        client.addPlayerMovedListener { event ->
            scope.launch {
                handleEvent { predictAction(getCurrentGameState()) }
            }
        }

        client.addActorDeathListener { event ->
            scope.launch {
                handleEvent { predictAction(getCurrentGameState()) }
            }
        }

        client.addItemContainerChangedListener { event ->
            scope.launch {
                handleEvent { predictAction(getCurrentGameState()) }
            }
        }
    }

    private fun setupCommands() {
        client.addCommandListener("rl_reload") { args ->
            val modelPath = args.firstOrNull() ?: return@addCommandListener
            val success = reloadModel(modelPath)
            val message = if (success) "Model reloaded successfully." else "Failed to reload model."
            client.addChatMessage(ChatMessageType.GAMEMESSAGE, "", message, null)
        }

        client.addCommandListener("rl_status") {
            client.addChatMessage(ChatMessageType.GAMEMESSAGE, "", "Reinforcement Learning Plugin is running.", null)
        }
    }

    private suspend fun handleEvent(predict: suspend () -> Int) {
        try {
            val action = predict()
            withContext(Dispatchers.Main) {
                performAction(action)
            }
        } catch (e: Exception) {
            logger.error("Error handling event", e)
        }
    }

    private suspend fun predictAction(input: String): Int {
        return withContext(Dispatchers.Default) {
            try {
                val inputTensor = Tensor.fromBlob(input.toByteArray(), longArrayOf(1, input.length.toLong()))
                val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
                outputTensor.dataAsLongArray[0].toInt()
            } catch (e: Exception) {
                logger.error("Error predicting action", e)
                -1
            }
        }
    }

    private fun performAction(action: Int) {
        when (convertActionToGameAction(action)) {
            "attack" -> performAttack()
            "defend" -> performDefend()
            "move_north" -> movePlayer(WorldPoint(client.localPlayer.worldLocation.x, client.localPlayer.worldLocation.y + 1, client.localPlayer.worldLocation.plane))
            "move_south" -> movePlayer(WorldPoint(client.localPlayer.worldLocation.x, client.localPlayer.worldLocation.y - 1, client.localPlayer.worldLocation.plane))
            "move_east" -> movePlayer(WorldPoint(client.localPlayer.worldLocation.x + 1, client.localPlayer.worldLocation.y, client.localPlayer.worldLocation.plane))
            "move_west" -> movePlayer(WorldPoint(client.localPlayer.worldLocation.x - 1, client.localPlayer.worldLocation.y, client.localPlayer.worldLocation.plane))
            "idle" -> performIdle()
            else -> performIdle()
        }
    }

    private fun convertActionToGameAction(action: Int): String {
        return actionMap[action] ?: "idle"
    }

    private fun performAttack() {
        client.addChatMessage(ChatMessageType.GAMEMESSAGE, "", "Performing Attack", null)
    }

    private fun performDefend() {
        client.addChatMessage(ChatMessageType.GAMEMESSAGE, "", "Performing Defend", null)
    }

    private fun movePlayer(worldPoint: WorldPoint) {
        client.localPlayer.setWorldLocation(worldPoint)
        client.addChatMessage(ChatMessageType.GAMEMESSAGE, "", "Moving to $worldPoint", null)
    }

    private fun performIdle() {
        client.addChatMessage(ChatMessageType.GAMEMESSAGE, "", "Idling", null)
    }

    private fun getCurrentGameState(): String {
        val player = client.localPlayer
        val playerHealth
