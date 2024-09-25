package com.finlogik.back.forum.controllers;

import com.finlogik.back.forum.entities.VoteTopic;
import com.finlogik.back.forum.services.VoteTopicService;
import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@AllArgsConstructor
@RequestMapping("/voteTopic")
public class VoteTopicRestController {

    private VoteTopicService voteService;
    @PostMapping("/voteLike/{idTopic}/{userId}")
    public VoteTopic voteUserlike(@PathVariable("idTopic") Long idTopic,
                                  @PathVariable("userId") String userId){
        return voteService.voteUserlike(idTopic,userId);
    }
    @PostMapping("/voteDislike/{idTopic}/{userId}")
    public VoteTopic voteUserDislike(@PathVariable("idTopic") Long idTopic,
                                     @PathVariable("userId") String userId){
        return voteService.voteUserdislike(idTopic,userId);
    }

    @GetMapping("/status/{topicId}/{userId}")
    public ResponseEntity<String> getUserVoteStatus(
            @PathVariable Long topicId,
            @PathVariable String userId) {
        String status = voteService.getUserVoteStatus(topicId, userId);
        return ResponseEntity.ok(status);
    }

}
