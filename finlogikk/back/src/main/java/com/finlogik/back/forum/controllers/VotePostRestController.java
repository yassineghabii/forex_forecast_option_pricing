package com.finlogik.back.forum.controllers;

import com.finlogik.back.forum.entities.VotePost;
import com.finlogik.back.forum.services.VotePostService;
import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@AllArgsConstructor
@RequestMapping("/votePost")
public class VotePostRestController {

    private VotePostService voteService;
    @PostMapping("/voteLike/{idPost}/{userId}")
    public VotePost voteUserlike(@PathVariable("idPost") Long idPost,
                                  @PathVariable("userId") String userId){
        return voteService.voteUserlike(idPost,userId);
    }
    @PostMapping("/voteDislike/{idPost}/{userId}")
    public VotePost voteUserDislike(@PathVariable("idPost") Long idPost,
                                     @PathVariable("userId") String userId){
        return voteService.voteUserdislike(idPost,userId);
    }

    @GetMapping("/status/{PostId}/{userId}")
    public ResponseEntity<String> getUserVoteStatus(
            @PathVariable Long PostId,
            @PathVariable String userId) {
        String status = voteService.getUserVoteStatus(PostId, userId);
        return ResponseEntity.ok(status);
    }

}
