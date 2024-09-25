package com.finlogik.back.forum.controllers;

import com.finlogik.back.forum.entities.VoteComment;
import com.finlogik.back.forum.services.VoteCommentService;
import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@AllArgsConstructor
@RequestMapping("/voteComment")
public class VoteCommentRestController {

    private VoteCommentService voteService;
    @PostMapping("/voteLike/{idComment}/{userId}")
    public VoteComment voteUserlike(@PathVariable("idComment") Long idComment,
                                  @PathVariable("userId") String userId){
        return voteService.voteUserlike(idComment,userId);
    }
    @PostMapping("/voteDislike/{idComment}/{use}")
    public VoteComment voteUserDislike(@PathVariable("idComment") Long idComment,
                                     @PathVariable("userId") String userId){
        return voteService.voteUserdislike(idComment,userId);
    }

    @GetMapping("/status/{CommentId}/{userId}")
    public ResponseEntity<String> getUserVoteStatus(
            @PathVariable Long CommentId,
            @PathVariable String userId) {
        String status = voteService.getUserVoteStatus(CommentId, userId);
        return ResponseEntity.ok(status);
    }

}
